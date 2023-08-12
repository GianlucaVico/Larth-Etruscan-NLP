"""
Inference for Larth.
"""
import functools
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm import tqdm

import jax
import flax
from flax.training import common_utils
import orbax.checkpoint
from data_utils import clean_etruscan

import decode
import larth
from train_utils import DataLoader, TrainConfig, pad_examples, tohost
from train import predict_step, initialize_cache
from data_utils import make_dataloader
import sys

sys.path.append("../")
import Data

# Store compiled functions
_paraller_functions: Dict[str, Callable] = {}


def load_params(path: str) -> flax.core.FrozenDict:
    """Load the model"""
    # path: .../[step]/default
    state = orbax.checkpoint.PyTreeCheckpointer().restore(path)  # Dict
    return state["params"]


def _translate(
    dl: DataLoader,
    p_pred_step: Callable[
        [jax.Array, jax.Array, flax.core.FrozenDict, flax.core.FrozenDict, int, int],
        jax.Array,
    ],
    p_init_cache: Callable[[jax.Array, jax.Array], jax.Array],
    params: flax.core.FrozenDict,
    decode_source: Callable[[np.array], str],
    decode_target: Callable[[np.array], str],
    max_predict_length: int,
) -> Tuple[List[str], List[str]]:
    """Translates the `predict_ds` and calculates the BLEU score."""
    n_devices = jax.local_device_count()
    sources = []
    predictions = []

    for batch in tqdm(dl, total=len(dl)):
        cur_batch_size = batch["source_chars"].shape[0]
        if cur_batch_size % n_devices:
            padded_size = int(np.ceil(cur_batch_size / n_devices) * n_devices)
            batch = jax.tree_util.tree_map(
                lambda x: pad_examples(x, padded_size), batch
            )

        batch = common_utils.shard(batch)
        cache = p_init_cache(batch["source_chars"], batch["source_words"])
        predicted = p_pred_step(
            batch["source_chars"],
            batch["source_words"],
            params,
            cache,
            decode.EOS_ID,
            max_predict_length,
        )

        predicted = tohost(predicted)
        inputs = tohost(batch["source_chars"])

        # Iterate through non-padding examples of batch.
        for i, s in enumerate(predicted[:cur_batch_size]):
            sources.append(decode_source(inputs[i]))
            predictions.append(decode_target(s))

    return sources, predictions


def translate(
    input_text: List[str],
    params: flax.core.FrozenDict,
    source_tokenizer: Data.SentencePieceTokenizer,
    target_tokenizer: Data.SentencePieceTokenizer,
    batch_size: int,
    beam_size: int,
    max_len: int,
    model_config: larth.LarthTranslationConfig,
    *,
    clean_cache: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Translate Etruscan with Larth

    Args:
        input_text: Etruscan texts
        params: model parameters
        source_tokenizer: tokenizer for the input text
        target_tokenizer: tokenizer for the output text
        batch_size: batch size
        beam_size: number of beams for beam search decoding
        max_len: maximum sequence length
        model_config: configuration of the model
        clean_cache: recompile Jax functions (e.g., for a different model)
    """
    global _paraller_functions
    if clean_cache:
        _paraller_functions.clear()

    config = TrainConfig(batch_size=batch_size, beam_size=beam_size, cached=False)
    # Duplicate the model for each device (otherwise pmap won't run)
    params = flax.jax_utils.replicate(params)

    # Prepare the data
    input_text = [clean_etruscan(i) for i in input_text]
    chars, words = source_tokenizer.tokenize(input_text, align=True)

    dl = make_dataloader(
        input_text,
        chars,
        words,
        [[]] * len(input_text),
        [[]] * len(chars),
        [[]] * len(words),
        config,
        max_len,
    )

    # Prepare the configuration
    model_config = larth.LarthTranslationConfig.replace(model_config)
    model_config = model_config.replace(
        char_vocab_size=source_tokenizer.vocab_size(words=False),
        word_vocab_size=source_tokenizer.vocab_size(),
        out_char_vocab_size=target_tokenizer.vocab_size(words=False),
        out_word_vocab_size=target_tokenizer.vocab_size(),
        deterministic=True,
        decode=True,
    )

    p_init_cache = _paraller_functions.get("p_init_cache", None)
    if p_init_cache is None:
        p_init_cache = jax.pmap(
            functools.partial(
                initialize_cache,
                config=model_config,
            ),
            axis_name="batch",
        )
        _paraller_functions["p_init_cache"] = p_init_cache

    p_pred_step = _paraller_functions.get("p_pred_step", None)
    if p_pred_step is None:
        p_pred_step = jax.pmap(
            functools.partial(predict_step, config=model_config, beam_size=beam_size),
            axis_name="batch",
            static_broadcasted_argnums=(4, 5),
        )
        _paraller_functions["p_pred_step"] = p_pred_step

    # Tokenizer functions
    target_eos_id = target_tokenizer._sp_words.eos_id()
    source_eos_id = source_tokenizer._sp_chars.eos_id()

    def decode_target_tokens(toks):
        valid_toks = toks[: np.argmax(toks == target_eos_id) + 1].astype(np.int32)
        return target_tokenizer.detokenize(valid_toks)

    def decode_source_tokens(toks):
        valid_toks = toks[: np.argmax(toks == source_eos_id) + 1].astype(np.int32)
        return source_tokenizer.detokenize(valid_toks, word=False)

    return _translate(
        dl,
        p_pred_step,
        p_init_cache,
        params,
        decode_source_tokens,
        decode_target_tokens,
        model_config.max_len,
    )
