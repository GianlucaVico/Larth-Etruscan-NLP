"""
Adapted from https://github.com/google/flax/blob/main/examples/wmt/train.py

Train the model.
"""
import functools
import gc
import logging
import os
import psutil
from dataclasses import asdict
from typing import Callable, Dict, Tuple

import decode
import flax
import larth
import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import pandas as pd
import sacrebleu
from clu import metric_writers, periodic_actions
from data_utils import get_training_data
from flax import jax_utils
from flax.training import common_utils, orbax_utils, train_state
from tqdm import tqdm
from train_utils import (
    DataLoader,
    TrainConfig,
    compute_metrics,
    compute_weighted_cross_entropy,
    create_learning_rate_schedule,
    pad_examples,
    tohost,
    save_config,
)

# jax.config.update('jax_disable_jit', True)
jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_log_compiles', True)


def _print_device(_, device):
    print(device)


@jax.jit
def print_device(x: jax.Array) -> None:
    """For debug: print the device of a jax Array"""
    hcb.call(_print_device, x, call_with_device=True)


# Use legacy API with orbax backend
flax.config.update("flax_use_orbax_checkpointing", True)

BatchType = Dict[str, jax.Array]


def train_step(
    state: train_state.TrainState,
    batch: BatchType,
    config: larth.LarthTranslationConfig,
    learning_rate_fn: Callable[[int], float],
    label_smoothing: float = 0.0,
    dropout_rng: jax.random.PRNGKeyArray | None = None,
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Perform a single training step.

    Args:
        state: current train state
        batch: data batch
        config: model configuration
        learning_rate_fn: learning rate schedule
        label smooting:
        dropout_rng: jax rng

    Returns:
        new train state and metrics
    """
    train_keys = ["source_chars", "source_words", "target_chars", "target_words"]

    (source_chars, source_words, _, target_words) = (  # not used
        batch.get(k, None) for k in train_keys
    )

    weights = jnp.where(target_words > 0, 1, 0).astype(jnp.float32)
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        """loss function used for training."""
        logits = larth.LarthTranslation(config).apply(
            {"params": params},
            source_chars,
            source_words,
            target_words,
            rngs={"dropout": dropout_rng},
        )

        loss, weight_sum = compute_weighted_cross_entropy(
            logits, target_words, weights, label_smoothing
        )
        mean_loss = loss / weight_sum
        return mean_loss, logits

    step = state.step
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, target_words, weights)
    metrics["learning_rate"] = learning_rate_fn(step)
    return new_state, metrics


def eval_step(
    params: flax.core.FrozenDict,
    batch: BatchType,
    config: larth.LarthTranslationConfig,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics on a batch.

    Args:
        params: model weights
        batch: data batch
        config: model configuration
        label_smoothing: smoothin factor

    Returns:
        Metrics as dictionary
    """
    source_chars, source_words, targets = (
        batch["source_chars"],
        batch["source_words"],
        batch["target_words"],
    )
    weights = jnp.where(targets > 0, 1.0, 0.0)
    logits = larth.LarthTranslation(config=config).apply(
        {"params": params}, source_chars, source_words, targets
    )

    return compute_metrics(logits, targets, weights, label_smoothing)


def initialize_cache(
    source_chars: jax.Array,
    source_words: jax.Array,
    config: larth.LarthTranslationConfig,
) -> flax.core.FrozenDict:
    """Initialize a cache for a given input shape and max decode length."""
    target_shape = (source_words.shape[0], config.max_len) + source_words.shape[2:]
    initial_variables = larth.LarthTranslation(config).init(
        jax.random.PRNGKey(0),
        jnp.ones(source_chars.shape, jnp.int32),
        jnp.ones(source_words.shape, jnp.int32),
        jnp.ones(target_shape, jnp.int32),
    )
    return initial_variables["cache"]


def predict_step(
    source_chars: jax.Array,
    source_words: jax.Array,
    params: flax.core.FrozenDict,
    cache: flax.core.FrozenDict,
    eos_id: int,
    max_decode_len: int,
    config: larth.LarthTranslationConfig,
    beam_size: int = 4,
    length_penalty: float = 0.6,
) -> jax.Array:
    """
    Predict translation with fast decoding beam search on a batch.

    Args:
        source_chars: batch of char sequences
        source_words: batch of word sequences
        params: model weights
        cache: cache from previous run
        eos_id: end of sequence token
        max_decode_len: maximum length of generated sequences
        config: model configuration
        beam_size: number of beams for decoding
        length_penalty: beam search length penalty

    Returns:
        Tokenized predicted sequences
    """
    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * beam_size, where each batch item"s data is expanded in-place
    # rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]

    encoded_inputs = decode.flat_batch_beam_expand(
        larth.LarthTranslation(config).apply(
            {"params": params},
            source_chars,
            source_words,
            method=larth.LarthTranslation.encode,
        ),
        beam_size,
    )
    raw_inputs = decode.flat_batch_beam_expand(source_words, beam_size)

    def tokens_ids_to_logits(flat_ids, flat_cache):
        """Token slice to logits from decoder model."""
        # --> [batch * beam, 1, vocab]
        flat_logits, new_vars = larth.LarthTranslation(config).apply(
            {"params": params, "cache": flat_cache},
            encoded_inputs,
            flat_ids,
            raw_inputs,  # only needed for input padding mask
            mutable=["cache"],
            method=larth.LarthTranslation.decode,
        )

        new_flat_cache = new_vars["cache"]
        # Remove singleton sequence-length dimension:
        # [batch * beam, 1, vocab] --> [batch * beam, vocab]
        flat_logits = flat_logits.squeeze(axis=1)
        return flat_logits, new_flat_cache

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    beam_seqs, _ = decode.beam_search(
        source_chars.shape[0],
        max_decode_len,
        cache,
        tokens_ids_to_logits,
        beam_size=beam_size,
        alpha=length_penalty,
        eos_id=eos_id,
    )
    # jax.debug.breakpoint()
    # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
    # sorted in increasing order of log-probability.
    # Return the highest scoring beam sequence, drop first dummy 0 token.
    return beam_seqs[:, -1, 1:]


def evaluate(
    *,
    p_eval_step: Callable[[flax.core.FrozenDict, BatchType], Dict[str, float]],
    params: flax.core.FrozenDict,
    eval_ds: DataLoader,
) -> Dict[str, float]:
    """
    Evaluate the model an return a dictionary with the metrics.

    Args:
        p_eval_step: p-mapped evaluation function (one step)
        params: model weights
        eval_ds: evaluation dataset

    Returns:
        Dictionary metrics
    """
    logging.info("Gathering evaluation metrics.")
    eval_metrics = []

    eval_iter = iter(eval_ds)
    for eval_batch in tqdm(
        eval_iter, total=len(eval_ds), leave=False, desc="Evaluation"
    ):
        eval_batch = common_utils.shard(eval_batch)
        metrics = p_eval_step(params, eval_batch)
        eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_util.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop("denominator")
    eval_summary = jax.tree_util.tree_map(
        lambda x: x / eval_denominator,
        eval_metrics_sums,
    )
    return eval_summary


def translate_and_calculate_bleu(
    *,
    p_pred_step: Callable[
        [jax.Array, jax.Array, flax.core.FrozenDict, flax.core.FrozenDict, int, int],
        jax.Array,
    ],
    p_init_cache: Callable[[jax.Array, jax.Array], jax.Array],
    params: flax.core.FrozenDict,
    predict_ds: DataLoader,
    decode_source: Callable[[np.ndarray], str],
    decode_target: Callable[[np.ndarray], str],
    eos_id: int,
    max_predict_length: int,
    train_ds: DataLoader | None = None,
) -> (
    Tuple[pd.DataFrame, Dict[str, float]]
    | Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]
):
    """
    Translates the `predict_ds` and calculates the BLEU score.

    Args:
        p_pred_step: p-mapped prediction function (one batch)
        p_init_cache: p-mapped function to initialize the model
        predict_ds: evaluation dataloader
        decode_source: function to decode the input character sequence
        decode_target: function to decode the output word sequence
        eos_id: end of sequence token
        max_predict_length: maximum length of the generated sequences
        train_ds: train dataloader

    Returns:
        Pandas dataframe with examples from the test set;
        if train_ds is given, pandas dataframe with examples from the train set;
        dictionary with the metrics

    Note: we decode the character sequence even if the model uses only the word sequence.
        In this way it is easy and the dataloader generate this sequence anyway-
    """
    n_devices = jax.local_device_count()

    logging.info("Translating evaluation dataset.")

    sources, references, predictions = [], [], []
    for pred_batch in tqdm(
        predict_ds, total=len(predict_ds), leave=False, desc="Prediction"
    ):
        cur_pred_batch_size = pred_batch["source_chars"].shape[0]
        if cur_pred_batch_size % n_devices:
            padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
            pred_batch = jax.tree_util.tree_map(
                lambda x: pad_examples(x, padded_size), pred_batch
            )

        pred_batch = common_utils.shard(pred_batch)
        cache = p_init_cache(pred_batch["source_chars"], pred_batch["source_words"])
        predicted = p_pred_step(
            pred_batch["source_chars"],
            pred_batch["source_words"],
            params,
            cache,
            eos_id,
            max_predict_length,
        )

        predicted = tohost(predicted)
        inputs = tohost(pred_batch["source_chars"])
        targets = tohost(pred_batch["target_words"])

        # Iterate through non-padding examples of batch.
        for i, s in enumerate(predicted[:cur_pred_batch_size]):
            sources.append(decode_source(inputs[i]))
            references.append(decode_target(targets[i]))
            predictions.append(decode_target(s))

    logging.info(
        f"Translation: {len(predictions)} predictions, {len(references)} references, {len(sources)} sources."
    )

    # Calculate BLEU score for translated eval corpus against reference.
    scores = {
        "bleu": sacrebleu.compat.corpus_bleu(
            predictions, [references], lowercase=True
        ).score,
        "chrf": sacrebleu.compat.corpus_chrf(predictions, [references]).score,
        "ter": sacrebleu.corpus_ter(
            predictions,
            [references],
            normalized=True,
            no_punct=True,
            case_sensitive=False,
        ).score,
    }

    # Save translation samples for tensorboard.
    test_examples = []
    for n in np.random.choice(np.arange(len(predictions)), 8):
        test_examples.append((sources[n], references[n], predictions[n]))
    test_examples = pd.DataFrame(test_examples, columns=["Source", "Ref.", "Pred."])

    if train_ds is not None:
        train_batch = next(iter(train_ds))
        train_batch = common_utils.shard(train_batch)
        cache = p_init_cache(train_batch["source_chars"], train_batch["source_words"])
        predicted = p_pred_step(
            train_batch["source_chars"],
            train_batch["source_words"],
            params,
            cache,
            decode.EOS_ID,
            max_predict_length,
        )
        predicted = tohost(predicted)
        inputs = tohost(train_batch["source_chars"])
        targets = tohost(train_batch["target_words"])

        train_examples = []
        for i, s in enumerate(predicted):
            train_examples.append(
                (decode_source(inputs[i]), decode_target(targets[i]), decode_target(s))
            )
        train_examples = pd.DataFrame(
            train_examples, columns=["Source", "Ref.", "Pred."]
        )
        return test_examples, train_examples, scores
    return test_examples, scores


def train_and_evaluate(
    model_config: larth.LarthTranslationConfig,
    run_config: TrainConfig,
) -> None:
    """
    Runs a training and evaluation loop.

    Args:
        model_config: model configuration to use.
        run_config: training configuration
    """
    try:
        os.makedirs(run_config.workdir)
    except FileExistsError:
        logging.info("Workdir already exists: skipping")

    logging.info("Saving configurations.")
    if run_config.train:
        save_config(
            os.path.join(
                run_config.workdir, f"{os.path.dirname(run_config.workdir)}_train.yml"
            ),
            asdict(run_config),
        )
        save_config(
            os.path.join(
                run_config.workdir, f"{os.path.dirname(run_config.workdir)}_model.yml"
            ),
            asdict(model_config),
        )

    # Load Dataset
    # ---------------------------------------------------------------------------
    logging.info("Initializing dataset.")
    (
        train_ds,
        test_ds,
        source_tokenizer,
        target_tokenizer,
    ) = get_training_data(
        run_config, model_config.max_len
    )  # , model_config.encoder_type)
    logging.info(f"Train batches: {len(train_ds)}")
    logging.info(f"Test batches: {len(test_ds)}")

    target_eos_id = target_tokenizer._sp_words.eos_id()
    source_eos_id = source_tokenizer._sp_chars.eos_id()

    def decode_target_tokens(toks: np.array) -> str:
        toks = np.append(toks, target_eos_id)
        valid_toks = toks[: np.argmax(toks == target_eos_id) + 1].astype(np.int32)
        return target_tokenizer.detokenize(valid_toks)

    def decode_source_tokens(toks: np.array) -> str:
        toks = np.append(toks, source_eos_id)  # Otherwise valid_toks is empty
        valid_toks = toks[: np.argmax(toks == source_eos_id) + 1].astype(np.int32)
        return source_tokenizer.detokenize(valid_toks, word=False)

    logging.info("Initializing model, optimizer, and step functions.")

    # Build Model and Optimizer
    # ---------------------------------------------------------------------------
    train_config = larth.LarthTranslationConfig.replace(model_config)
    train_config = train_config.replace(
        char_vocab_size=source_tokenizer.vocab_size(words=False),
        word_vocab_size=source_tokenizer.vocab_size(),
        out_char_vocab_size=target_tokenizer.vocab_size(words=False),
        out_word_vocab_size=target_tokenizer.vocab_size(),
        deterministic=False,
    )
    eval_config = train_config.replace(deterministic=True)
    predict_config = train_config.replace(deterministic=True, decode=True)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    input_shape = (run_config.batch_size, train_config.max_len)
    target_shape = (run_config.batch_size, train_config.max_len)

    model = larth.LarthTranslation(config=eval_config)

    initial_variables = jax.jit(model.init)(
        init_rng,
        chars=jnp.ones(input_shape, jnp.int32),
        words=jnp.ones(input_shape, jnp.int32),
        targets=jnp.ones(target_shape, jnp.int32),
    )
    # Create train state with Adam optimizer and weight decay.
    logging.info("Initializing learning rate")
    learning_rate_fn = create_learning_rate_schedule(
        lr=run_config.lr, warmup_steps=run_config.warmup_steps
    )
    # learning_rate_fn = lambda x: run_config.lr
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=initial_variables["params"],
        # tx=optax.adamw(
        #     learning_rate=learning_rate_fn,
        #     b1=0.9,
        #     b2=0.98,
        #     eps=1e-9,
        #     weight_decay=run_config.weight_decay,
        # ),
        tx=optax.radam(
            learning_rate=learning_rate_fn,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
        ),
    )

    del initial_variables

    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpointer = orbax.checkpoint.Checkpointer(
        orbax.checkpoint.PyTreeCheckpointHandler()
    )
    mngr = orbax.checkpoint.CheckpointManager(run_config.workdir, checkpointer, options)
    start_step = 0
    start_epoch = 0
    if run_config.restore_checkpoints:
        logging.info("Restoring checkpoint")
        try:
            if run_config.restore_from is None:
                start_epoch = mngr.latest_step()
                state = mngr.restore(start_epoch, items=state)
                start_epoch += 1
                start_step = start_epoch * len(train_ds)
            else:
                state = orbax.checkpoint.PyTreeCheckpointer().restore(
                    run_config.restore_from, item=state
                )
            print(f"Resuming from {start_epoch}")
        except FileNotFoundError:
            start_step = 0
            start_epoch = 0
            logging.info("Checkpoint does not exist. Skipping")

    logging.info("Creating metric writers")
    writer = metric_writers.create_default_writer(
        run_config.workdir, just_logging=jax.process_index() > 0
    )
    if start_step == 0:
        tmp = {}
        tmp.update(asdict(run_config))
        tmp.update(asdict(train_config))
        writer.write_hparams(tmp)

    # Replicate state.
    state = jax_utils.replicate(state)

    # Compile multidevice versions of train/eval/predict step and cache init fn.
    logging.info("Compiling multidevice fn")
    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            config=train_config,
            learning_rate_fn=learning_rate_fn,
            label_smoothing=run_config.label_smoothing,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )
    p_eval_step = jax.pmap(
        functools.partial(eval_step, config=eval_config), axis_name="batch"
    )
    p_init_cache = jax.pmap(
        functools.partial(
            initialize_cache,
            # max_decode_len=predict_config.max_len,
            config=predict_config,
        ),
        axis_name="batch",
    )
    p_pred_step = jax.pmap(
        functools.partial(
            predict_step,
            config=predict_config,
            beam_size=run_config.beam_size,
            length_penalty=run_config.length_penalty,
        ),
        axis_name="batch",
        static_broadcasted_argnums=(4, 5),
    )  # eos token, max_length are constant

    # Main Train Loop
    # ---------------------------------------------------------------------------

    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap"d training update for performance.
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    del rng

    logging.info("Starting training loop.")
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=run_config.epochs, writer=writer
    )
    # if jax.process_index() == 0:
    #     hooks += [
    #         report_progress,
    #         periodic_actions.Profile(logdir=run_config.workdir, num_profile_steps=5), # This fill all the RAM
    #     ]
    train_metrics = []
    gc.collect()
    with metric_writers.ensure_flushes(writer):
        step = start_step
        if run_config.train:
            for epoch in tqdm(range(start_epoch, run_config.epochs), desc="Epoch"):
                # print("Epoch:", epoch, end=None)
                train_iter = iter(train_ds)
                for batch in tqdm(
                    train_iter, total=len(train_ds), leave=False, desc="Step"
                ):
                    # for batch in train_iter:
                    step += 1
                    logging.info(f"Step: {step}")

                    # Shard data to devices and do a training step.
                    with jax.profiler.StepTraceAnnotation("train", step_num=step):
                        batch = common_utils.shard(
                            jax.tree_util.tree_map(jnp.asarray, batch)
                        )
                        state, metrics = p_train_step(
                            state, batch, dropout_rng=dropout_rngs
                        )
                        train_metrics.append(metrics)

                gc.collect()
                # Quick indication that training is happening.
                logging.info(f"Finished training epoch {epoch + 1}.")
                # for h in hooks:
                #     h(epoch)
                # writer.write_scalars(epoch, {"RAM": psutil.virtual_memory().percent})
                writer.write_scalars(epoch, {"RAM": psutil.Process().memory_percent()})

                # # Evaluate after every epoch and at last epoch
                is_eval_epoch = ((epoch + 1) == run_config.epochs) or (
                    ((epoch + 1) % run_config.eval_every_epochs) == 0
                )
                if is_eval_epoch and run_config.eval:
                    logging.info("Evaluating")
                    with report_progress.timed("training_metrics"):
                        logging.info("Gathering training metrics.")
                        train_metrics = common_utils.get_metrics(train_metrics)
                        lr = train_metrics.pop("learning_rate").mean()
                        metrics_sums = jax.tree_util.tree_map(jnp.sum, train_metrics)
                        denominator = metrics_sums.pop("denominator")
                        summary = jax.tree_util.tree_map(
                            lambda x: x / denominator, metrics_sums
                        )
                        summary["learning_rate"] = lr
                        summary = {"train_" + k: v for k, v in summary.items()}
                        writer.write_scalars(epoch, summary)
                        train_metrics = []

                    with report_progress.timed("eval"):
                        eval_results = evaluate(
                            p_eval_step=p_eval_step,
                            params=state.params,
                            eval_ds=test_ds,
                        )
                        writer.write_scalars(
                            epoch, {"eval_" + k: v for k, v in eval_results.items()}
                        )

                    with report_progress.timed("translate_and_bleu"):
                        (
                            examples,
                            train_examples,
                            bleu_score,
                        ) = translate_and_calculate_bleu(
                            p_pred_step=p_pred_step,
                            p_init_cache=p_init_cache,
                            params=state.params,
                            predict_ds=test_ds,
                            decode_source=decode_source_tokens,  # Can't decode words with alignment
                            decode_target=decode_target_tokens,
                            eos_id=target_tokenizer._sp_words.eos_id(),
                            max_predict_length=int(predict_config.max_len),
                            train_ds=train_ds,
                        )

                        writer.write_scalars(epoch, bleu_score)
                        writer.write_texts(
                            epoch,
                            {
                                "samples": examples.to_markdown(),
                                "train_samples": train_examples.to_markdown(),
                            },
                        )
                # Save a checkpoint on one host after every checkpoint_freq steps.
                if (epoch + 1) % run_config.checkpoint_every_epochs == 0:
                    logging.info(f"Saving checkpoint epoch {epoch}")
                    with report_progress.timed("checkpoint"):
                        ckpt = jax_utils.unreplicate(state)
                        save_args = orbax_utils.save_args_from_target(ckpt)
                        mngr.save(epoch, ckpt, save_kwargs={"save_args": save_args})
        elif run_config.eval:
            logging.info("Evaluating")
            eval_results = evaluate(
                p_eval_step=p_eval_step,
                params=state.params,
                eval_ds=test_ds,
            )
            print("Evaluation scores")
            for k, v in eval_results.items():
                print(f"{k}: {v}")

            examples, bleu_score = translate_and_calculate_bleu(
                p_pred_step=p_pred_step,
                p_init_cache=p_init_cache,
                params=state.params,
                predict_ds=test_ds,
                decode_source=decode_source_tokens,  # Can't decode words with alignment
                decode_target=decode_target_tokens,
                eos_id=target_tokenizer._sp_words.eos_id(),
                max_predict_length=int(predict_config.max_len),
            )
            print("Metrics")
            for k, v in bleu_score.items():
                print(f"{k}: {v}")
            print(examples)
