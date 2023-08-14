"""
Adapted from https://github.com/google/flax/tree/main/examples/lm1b and Larth translation.
"""
import functools
import gc
import logging
import os
import psutil
from dataclasses import asdict
from typing import Dict, Callable, Tuple

import flax
import LMLarth
import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import orbax.checkpoint
from clu import metric_writers, periodic_actions
from data_utils import get_training_data
from flax import jax_utils
from flax.training import common_utils, orbax_utils, train_state
from tqdm import tqdm, trange
import sklearn.metrics

from train_utils import (
    compute_weighted_cross_entropy,
    compute_metrics,
    tohost,
    pad_examples,
    save_config,
    DataLoader,
    TrainConfig,
    create_learning_rate_schedule,
    PPL,
)
import sys

sys.path.append("../")
import Translation.Larth.decode as decode
import temperature

# jax.config.update('jax_disable_jit', True)
# jax.config.update("jax_debug_nans", True)
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
    config: LMLarth.LarthLMConfig,
    learning_rate_fn: Callable[[int], float],
    label_smoothing: float = 0.0,
    dropout_rng=None,
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
    # train_keys = ["inputs"]
    # (inputs) = (batch.get(k, None) for k in train_keys,)

    inputs = batch.get("inputs")
    weights = jnp.where(inputs > 0, 1, 0).astype(jnp.float32)
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        """loss function used for training."""
        logits = LMLarth.LarthLM(config).apply(
            {"params": params}, inputs, rngs={"dropout": dropout_rng}
        )

        loss, weight_sum = compute_weighted_cross_entropy(
            logits, inputs, weights, label_smoothing
        )
        mean_loss = loss / weight_sum
        return mean_loss, logits

    step = state.step
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, inputs, weights)
    metrics["learning_rate"] = learning_rate_fn(step)
    return new_state, metrics


def eval_step(
    params: flax.core.FrozenDict,
    batch: BatchType,
    config: LMLarth.LarthLMConfig,
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
    inputs = batch["inputs"]
    weights = jnp.where(inputs > 0, 1.0, 0.0)
    logits = LMLarth.LarthLM(config).apply({"params": params}, inputs)

    return compute_metrics(logits, inputs, weights, label_smoothing)


def decode_beam_search(
    inputs: jax.Array,
    cache: jax.Array,
    tokens_ids_to_logits: Callable[[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]],
    max_decode_len: int,
    beam_size: int,
    length_penalty: float,
    eos_id: int,
) -> jax.Array:
    """
    Beam search decoding

    Args:
        inputs: input batch
        cache: model cache
        tokens_ids_to_logits: call the model and return the new output and cache
        max_decode_len: maximun length of the decode sequences
        length_penalty: beam search length penalty
        eos_id: end-of-sequence token

    Returns:
        Decoded sequence
    """
    beam_seqs, _ = decode.beam_search(
        inputs.shape[0],
        max_decode_len,
        cache,
        tokens_ids_to_logits,
        beam_size=beam_size,
        alpha=length_penalty,
        eos_id=eos_id,
    )
    return beam_seqs[:, -1, 1:]


def decode_temperature(
    inputs: jax.Array,
    cache: jax.Array,
    tokens_ids_to_logits,
    rngkey: jax.random.KeyArray,
    temp: float,
    topk: int,
    eos_id: int,
) -> jax.Array:
    """
    Top-k decoding

    Args:
        inputs: input batch
        cache: model cache
        tokens_ids_to_logits: call the model and return the new output and cache
        rngkey: jax rng
        temp: temperature
        topk: k in top-k
        eos_id: end-of-sequence token

    Returns:
        Decoded sequence
    """
    seqs = temperature.temperature_sample(
        inputs,
        cache,
        tokens_ids_to_logits,
        rngkey,
        temperature=temp,
        topk=topk,
        eos_token=eos_id,
    )

    return seqs


def predict_step(
    inputs: jax.Array,
    params: flax.core.FrozenDict,
    max_decode_len: int,
    config: LMLarth.LarthLMConfig,
    decode_fn: Callable,
) -> jax.Array:
    """
    Predict lm on a batch.

    Args:
        inputs: input batch
        params: model weights
        cache: cache from previous run
        max_decode_len: maximum length of generated sequences
        config: model configuration
        decode_fn: decoding function (`decode_beam_search` or `decode_temperature`)

    Returns:
        Tokenized predicted sequences
    """
    target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
    initial_variables = LMLarth.LarthLM(config).init(
        jax.random.PRNGKey(0), jnp.ones(target_shape, config.dtype)
    )
    cache = initial_variables["cache"]

    def tokens_ids_to_logits(flat_ids, flat_cache):
        """Token slice to logits from decoder model."""
        # jax.debug.breakpoint()
        flat_logits, new_vars = LMLarth.LarthLM(config).apply(
            {"params": params, "cache": flat_cache},
            flat_ids,
            mutable=["cache"],
            # method=LMLarth.LarthLM.__call__
        )

        new_flat_cache = new_vars["cache"]
        # Remove singleton sequence-length dimension:
        # [batch, 1, vocab] --> [batch, vocab]
        flat_logits = flat_logits.squeeze(axis=1)
        return flat_logits, new_flat_cache

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    # jax.debug.breakpoint()
    return decode_fn(inputs, cache, tokens_ids_to_logits)


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
        lambda x: x / eval_denominator, eval_metrics_sums
    )
    return eval_summary


def generate_prediction(
    *,
    p_pred_step: Callable[[flax.core.FrozenDict, BatchType], Dict[str, float]],
    params: flax.core.FrozenDict,
    pred_batch: BatchType,
    decode_tokens: Callable[[np.ndarray], str],
    prompt_len: int = 4,
    eos_id: int = 2,
) -> pd.DataFrame:
    """
    Generate text from the prompt.

    Args:
        p_pred_step: p-mapped prediction function (one batch)
        params: model weights
        pred_batch: test batch
        decode_tokens: function to decode the tokens into strings
        prompt_len: use the first prompt_len tokens as input for the model
        eos_id: end of sequence token

    Returns:
        Pandas dataframe with predictions.
    """
    n_devices = jax.local_device_count()

    logging.info("Generating text.")
    predictions = []
    originals = []
    # Use batch of prompts provided by user.
    # pred_batch = next(iter(test_ds))
    pred_batch = pred_batch["inputs"]
    # Use only the first few tokens for the predictions
    new_pred_batch = pred_batch.at[:, prompt_len:].set(0)
    # Remove eos
    new_pred_batch = new_pred_batch.at[new_pred_batch == eos_id].set(0)

    cur_pred_batch_size = new_pred_batch.shape[0]
    if cur_pred_batch_size % n_devices:
        padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
        new_pred_batch = jax.tree_util.tree_map(
            lambda x: pad_examples(x, padded_size), new_pred_batch
        )
    new_pred_batch = common_utils.shard(new_pred_batch)

    # jax.debug.breakpoint()
    predicted = p_pred_step(
        new_pred_batch,
        params,
    )

    predicted = tohost(predicted)
    new_pred_batch = tohost(new_pred_batch)
    # Iterate through non-padding examples of batch.
    for s, p in zip(predicted[:cur_pred_batch_size], pred_batch[:cur_pred_batch_size]):
        prediction = decode_tokens(s)
        logging.info(f"Sample: {str(prediction)}")
        predictions.append(prediction)
        originals.append(decode_tokens(p))
    examples = pd.DataFrame({"Originals": originals, "Predictions": predictions})
    return examples


def eval_restoration(
    *,
    p_pred_step: Callable[[flax.core.FrozenDict, BatchType], Dict[str, float]],
    params: flax.core.FrozenDict,
    eval_ds: DataLoader,
    decode_tokens: Callable[[np.ndarray], str],
    eos_id: int = 2,
) -> Dict[str, float]:
    """
    Test the model on the restoration task.

    Args:
        p_pred_step: p-mapped prediction function (one batch)
        params: model weights
        eval_ds: test dataloader
        decode_tokens: function to decode the tokens into strings
        eos_id: end of sequence token

    Returns:
        Dictionary with the metrics
    """
    # Generate texts one token at the time and compare
    # Similar to generate_prediction but more inefficient
    # Ignore pad
    logging.info("Restoration")
    n_devices = jax.local_device_count()

    # Use only the first few tokens for the predictions
    predicted_tokens = []
    original_tokens = []

    eval_iter = iter(eval_ds)
    # mask = flax.linen.make_causal_mask(next(iter(eval_ds))["inputs"], dtype="int32")
    for pred_batch in tqdm(
        eval_iter, total=len(eval_ds), leave=False, desc="Restoration"
    ):
        pred_batch = pred_batch["inputs"]
        for t in trange(3, pred_batch.shape[1] - 1, leave=False):
            cur_pred_batch_size = pred_batch.shape[0]
            if cur_pred_batch_size % n_devices:
                padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
                pred_batch = jax.tree_util.tree_map(
                    lambda x: pad_examples(x, padded_size), pred_batch
                )
            mask = flax.linen.make_causal_mask(pred_batch, dtype="int32")
            masked_pred_batch = (
                pred_batch * mask[:, 0, t - 1]
            )  # Mask starts from [1, 0, 0, ...]
            # Remove eos
            masked_pred_batch = masked_pred_batch * (masked_pred_batch != eos_id)

            masked_pred_batch = common_utils.shard(masked_pred_batch)

            predicted = p_pred_step(
                masked_pred_batch,
                params,
            )
            predicted = tohost(predicted)
            masked_pred_batch = tohost(masked_pred_batch)
            # Remove pad
            tmp = zip(
                predicted[:cur_pred_batch_size, t].tolist(),
                pred_batch[:cur_pred_batch_size, t].tolist(),
            )
            tmp = [(i, j) for i, j in tmp if j != 0]
            if len(tmp) != 0:
                pred, orig = zip(*tmp)
                predicted_tokens.extend(pred)
                original_tokens.extend(orig)

    predicted_tokens_all = [decode_tokens(i) for i in predicted_tokens]
    original_tokens_all = [decode_tokens(i) for i in original_tokens]

    # Scores when the target word is known and exclude spaces
    # ▁ is decoded as ''
    # HACK: ▁'s id is 4, ids 0,1,2,3,4 are not words
    # tmp = [(i, j) for i, j in zip(original_tokens, predicted_tokens) if "-" not in i and "▁" not in i]
    tmp = [(i, j) for i, j in zip(original_tokens, predicted_tokens) if j > 4]
    tmp = [(decode_tokens(i), decode_tokens(j)) for i, j in tmp]
    tmp = [(i, j) for i, j in tmp if "-" not in i]
    scores = {}
    if len(tmp) != 0:
        original_tokens_clean, predicted_tokens_clean = zip(*tmp)
        scores["Accuracy"] = sklearn.metrics.accuracy_score(
            original_tokens_all, predicted_tokens_all
        )
        scores["Precision"] = sklearn.metrics.precision_score(
            original_tokens_all, predicted_tokens_all, average="macro"
        )
        scores["Recall"] = sklearn.metrics.recall_score(
            original_tokens_all, predicted_tokens_all, average="macro"
        )
        scores["F1-Score"] = sklearn.metrics.f1_score(
            original_tokens_all, predicted_tokens_all, average="macro"
        )

        scores["Accuracy_words"] = sklearn.metrics.accuracy_score(
            original_tokens_clean, predicted_tokens_clean
        )
        scores["Precision_words"] = sklearn.metrics.precision_score(
            original_tokens_clean, predicted_tokens_clean, average="macro"
        )
        scores["Recall_words"] = sklearn.metrics.recall_score(
            original_tokens_clean, predicted_tokens_clean, average="macro"
        )
        scores["F1-Score_words"] = sklearn.metrics.f1_score(
            original_tokens_clean, predicted_tokens_clean, average="macro"
        )
    return scores


def train_and_evaluate(
    model_config: LMLarth.LarthLMConfig, run_config: TrainConfig
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
        base_name = os.path.dirname(run_config.workdir)
        if base_name is None or base_name == "":
            base_name = os.path.basename(run_config.workdir)
        save_config(
            os.path.join(run_config.workdir, f"{base_name}_train.yml"),
            asdict(run_config),
        )
        save_config(
            os.path.join(run_config.workdir, f"{base_name}_model.yml"),
            asdict(model_config),
        )

    # Load Dataset
    # ---------------------------------------------------------------------------
    logging.info("Initializing dataset.")
    logging.info("Initializing dataset.")
    (
        train_ds,
        test_ds,
        tokenizer,
    ) = get_training_data(run_config, model_config.max_len)

    logging.info(f"Train batches: {len(train_ds)}")
    logging.info(f"Test batches: {len(test_ds)}")

    eos_id = tokenizer._sp_words.eos_id()

    def decode_tokens(toks):
        toks = np.append(toks, eos_id)
        valid_toks = toks[: np.argmax(toks == eos_id) + 1].astype(np.int32)
        return tokenizer.detokenize(valid_toks)

    logging.info("Initializing model, optimizer, and step functions.")
    # Build Model and Optimizer
    # ---------------------------------------------------------------------------
    train_config = LMLarth.LarthLMConfig.replace(model_config)
    train_config = train_config.replace(
        vocab_size=tokenizer.vocab_size(),
        deterministic=False,
        decode=False,
    )
    eval_config = train_config.replace(deterministic=True)
    predict_config = train_config.replace(deterministic=True, decode=True)

    start_step = 0
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    rng, inference_rng = jax.random.split(rng)
    input_shape = (run_config.batch_size, train_config.max_len)

    model = LMLarth.LarthLM(eval_config)
    initial_variables = jax.jit(model.init)(init_rng, jnp.ones(input_shape, jnp.int32))

    logging.info("Initializing learning rate")
    learning_rate_fn = create_learning_rate_schedule(
        lr=run_config.lr, warmup_steps=run_config.warmup_steps
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=initial_variables["params"],
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

    # Replicate optimizer.
    state = jax_utils.replicate(state)

    # compile multidevice versions of train/eval/predict step fn.
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

    if run_config.use_topk:
        decode_fn = functools.partial(
            decode_temperature,
            rngkey=inference_rng,
            temp=run_config.temperature,
            topk=run_config.topk,
            eos_id=eos_id,
        )
    else:
        decode_fn = functools.partial(
            decode_beam_search,
            max_decode_len=model_config.max_len,
            beam_size=run_config.beam_size,
            length_penalty=run_config.length_penalty,
            eos_id=eos_id,
        )

    p_pred_step = jax.pmap(
        functools.partial(
            predict_step,
            config=predict_config,
            max_decode_len=model_config.max_len,
            decode_fn=decode_fn,
        ),
        axis_name="batch",
        # static_broadcasted_argnums=(3, 4),
    )  # eos token, max_length are constant

    # Main Train Loop
    # ---------------------------------------------------------------------------

    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap"d training update for performance.
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    del rng

    logging.info("Starting training loop.")
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=run_config.epochs, writer=writer
    )

    gc.collect()
    with metric_writers.ensure_flushes(writer):
        step = start_step
        if run_config.train:
            for epoch in tqdm(range(start_epoch, run_config.epochs), desc="Epoch"):
                # print("Epoch:", epoch, end=None)
                train_iter = iter(train_ds)
                train_metrics = []
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

                # Periodic metric handling.
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
                        summary["PPL"] = PPL(summary["loss"])
                        summary["learning_rate"] = lr
                        summary = {"train_" + k: v for k, v in summary.items()}
                        writer.write_scalars(epoch, summary)

                    with report_progress.timed("eval"):
                        eval_results = evaluate(
                            p_eval_step=p_eval_step,
                            params=state.params,
                            eval_ds=test_ds,
                        )
                        # (clipped) perplexity after averaging log-perplexities
                        # Log perplexity are computed as croos-entropy between the inputs and shifted inputs
                        eval_results["PPL"] = PPL(eval_results["loss"])

                        writer.write_scalars(
                            epoch, {"eval_" + k: v for k, v in eval_results.items()}
                        )

                    with report_progress.timed("generate_text"):
                        pred_batch = next(iter(test_ds))
                        examples = generate_prediction(
                            p_pred_step=p_pred_step,
                            params=state.params,
                            pred_batch=pred_batch,
                            decode_tokens=decode_tokens,
                            prompt_len=run_config.prompt_len,
                            eos_id=eos_id,
                        )
                        writer.write_texts(epoch, {"samples": examples.to_markdown()})

                    with report_progress.timed("restoration"):
                        restoration_scores = eval_restoration(
                            p_pred_step=p_pred_step,
                            params=state.params,
                            eval_ds=test_ds,
                            decode_tokens=decode_tokens,
                            eos_id=eos_id,
                        )
                        writer.write_scalars(
                            epoch,
                            {"eval_" + k: v for k, v in restoration_scores.items()},
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
            eval_results["perplexity"] = jnp.clip(
                jnp.exp(eval_results["loss"]), a_max=1.0e4
            )
            print("Evaluation scores")
            for k, v in eval_results.items():
                print(f"{k}: {v}")
            pred_batch = common_utils.shard(
                jax.tree_util.tree_map(jnp.asarray, next(iter(test_ds)))
            )
            examples = generate_prediction(
                p_pred_step=p_pred_step,
                params=state.params,
                pred_batch=pred_batch,
                decode_tokens=decode_tokens,
                prompt_len=run_config.prompt_len,
                eos_id=eos_id,
            )

            print(examples)

            restoration_scores = eval_restoration(
                p_pred_step=p_pred_step,
                params=state.params,
                eval_ds=test_ds,
                decode_tokens=decode_tokens,
                eos_id=eos_id,
            )
            print("Restoration:")
            for k, v in restoration_scores:
                print(f"{k}:", v)
