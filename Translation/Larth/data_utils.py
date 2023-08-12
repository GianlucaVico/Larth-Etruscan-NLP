"""
Functions to clean, augment and load the dataset for Larth
"""
import itertools
import logging
import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import unicodedata
from train_utils import DataLoader, TrainConfig

import sys

sys.path.append("../")
sys.path.append("../..")
import Data
import Data.utils as utils

### Augmentation ###


def name_augmentation(
    et: List[str], eng: List[str], vocab: str, max_replacements: int, seed: int
) -> Tuple[List[str], List[str]]:
    """
    Perform name augmentation on the data.

    Note: only for Etruscan-English

    Args:
        et: Etruscan strings
        eng: English strings
        vocab: path to the pos csv
        max_replacements: replace each name up to this number of times (see Data.generate_bi for more details)
        rng: rng

    Returns:
        List of Etruscan strings and list of English strings with additional examples
    """
    vocab = Data.load_pos(Data._dir + "ETP_POS.csv")
    vocab["Etruscan"] = vocab["Etruscan"].apply(
        lambda x: utils.replace(x, utils.to_latin)
    )

    # Name augmentation needs pandas dataframes
    train_doc = pd.DataFrame({"Etruscan": et, "Translation": eng})
    index, _, _ = Data.create_index(vocab)

    # This should return a list of pairs (etruscan, english)
    gen, _ = Data.generate_pairs(
        train_doc,
        index,
        max_replacements=(
            max_replacements,
            min(2, max_replacements // 2),
            min(1, max_replacements // 4),
        ),
        seed=seed,
    )
    original = set(zip(et, eng))
    new = set(gen).union(original)
    et, eng = zip(*new)
    return et, eng


def unk_augmentation(
    source: List[str],
    prob: float,
    length: float,
    iterations: int,
    rng: np.random.RandomState,
    mask: str = "-",
) -> List[str]:
    """
    Corrupt some characters.
    The changes are focused at the begging and ending of the words.
    Each end of a word is corrupt following a Bernoulli distribution.
    The number of corrupted characters follows a geometric distribution.
    token -> token
    token -> --ken
    token -> toke-
    token -> -ok--
    This method preserve the order.

    Args:
        source: list of strings
        prob: probability of corrupting one end of a token
        length: average length of a the corruption
        iterations: number of examples to generate from a single example
        rng: random number generator
        mask: string used for the corruption

    Returns:
        List with the original strings and the augmented strings
    """
    tokens = [i.split() for i in source]

    def augment(tokens):
        # Select which ends to corrupt
        start = [rng.binomial(1, prob, len(i)) for i in tokens]
        end = [rng.binomial(1, prob, len(i)) for i in tokens]

        # The mean of the geometric distribution is 1/p.
        len_prob = 1 / length
        start_len = [
            rng.geometric(len_prob, len(i)) * i for i in start
        ]  # Sample and mask
        end_len = [rng.geometric(len_prob, len(i)) * i for i in end]

        # Apply
        augmented = [None] * len(tokens)
        for i, (t, s, e) in enumerate(zip(tokens, start_len, end_len)):
            # Attach the corrupted string to the uncorrupted tail of the token
            tmp = [mask * k + j[k:] for j, k in zip(t, s)]

            # Attach the uncorrupted head of the token to the corrupted tail
            tmp = [j[: len(j) - k] + mask * k for j, k in zip(tmp, e)]

            augmented[i] = " ".join(tmp)

        return augmented

    augmented = [augment(tokens) for _ in range(iterations)]
    augmented = list(itertools.chain.from_iterable(augmented))

    augmented = list(source + augmented)
    return augmented


def augment(
    source: List[str],
    target: List[str],
    config: TrainConfig,
    rng: np.random.RandomState,
) -> Tuple[List[str], List[str]]:
    """
    Perform data augmentation according to the given configuration

    Args:
        source: list of Etruscan texts
        target: list of English translations
        config: parameters for the data augmentation
        rng: rng

    Returns:
        List of Etruscan text and list of English translations
    """
    if (
        config.name_augmentation_max_replacements != 0
        and not config.debug
        and config.source_lang == "etruscan"
    ):
        logging.info("Performing name augmentation")
        source, target = name_augmentation(
            source,
            target,
            config.etruscan_vocab,
            config.name_augmentation_max_replacements,
            config.seed,
        )

    if (
        config.unk_augmentation_prob != 0
        and config.unk_augmentation_iterations != 0
        and not config.debug
    ):
        logging.info("Performing unk augmentation")
        source = unk_augmentation(
            source,
            config.unk_augmentation_prob,
            config.unk_augmentation_len,
            config.unk_augmentation_iterations,
            rng,
        )
        target = target * config.unk_augmentation_iterations

    return source, target


### Cleaning ###

title_re = re.compile(r"[^a-zA-Z ]*((mr)|(ms)|(mrs)|(miss))[^a-zA-Z ]*")
remove_chars = re.compile(r"[126\[\],<>]")
split_dash = re.compile(
    r"(?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])"
)  # preceded by word and followed by word
space_norm = re.compile(r" +")
add_unk = re.compile(r"\?")
non_word = re.compile(r"[^a-zA-Z ]")


def clean_english(x: str) -> str:
    """Clean English string"""
    x = x.lower()
    x = title_re.sub(" ", x)
    x = split_dash.sub(" ", x)
    x = remove_chars.sub(" ", x)
    x = add_unk.sub(" ", x)  # Remove ? from training data -> ? to <unk>
    x = space_norm.sub(" ", x)
    return x.strip()


def clean_etruscan(x: str) -> str:
    """Clean Etruscan string"""
    x = x.lower()
    x = remove_chars.sub(" ", x)
    x = space_norm.sub(" ", x)
    return x.strip()


def clean_tatoeba(x: str) -> str:
    """Clean string from the Tatoeba dataset"""
    x = x.lower()

    # Split accents, remove and recombine
    x = unicodedata.normalize("NFD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = unicodedata.normalize("NFC", x)

    x = x.translate(utils.greek_to_latin)
    x = x.translate(utils.others)

    x = non_word.sub(" ", x)  # This also remove punctuation
    x = space_norm.sub(" ", x)
    x = x.strip()
    return x


### Other actions ###
def remove_invalid(source: List[str], target: List[str]) -> Tuple[List[str], List[str]]:
    """
    Remove empty strings

    Args:
        source: list of texts from the source language
        target: list of translations from the target language

    Returns:
        List of source texts and list of target texts
    """
    assert len(source) == len(target)
    tmp = set(zip(source, target))
    tmp = [(i, j) for i, j in tmp if len(i.strip()) != 0 and len(j.strip()) != 0]
    source, target = zip(*tmp)
    return source, target


def select_debug(
    config: TrainConfig,
    train_source: List[str],
    train_target: List[str],
    test_source: List[str],
    test_target: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Use only few example for debugging"""
    if config.debug:
        train_source = train_source[: min(config.batch_size * 2, len(train_source))]
        test_source = test_source[: min(config.batch_size * 2, len(test_source))]
        train_target = train_target[: min(config.batch_size * 2, len(train_target))]
        test_target = test_target[: min(config.batch_size * 2, len(test_target))]
    return train_source, train_target, test_source, test_target


def split(
    source: List[str],
    target: List[str],
    config: TrainConfig,
    rng: np.random.RandomState,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split the dataset in train and test.

    Args:
        source: source texts
        target: target translations
        config: parameters for the split
        rng: rng

    Returns:
        Tuple with the list of train source text, train target translations,
            test source text, and test target translations.
    """
    assert len(source) == len(target)

    # Select the train samples
    train_indexes = rng.choice(
        len(source), int(np.ceil(len(source) * config.train_size)), replace=False
    )

    is_train = np.zeros(len(source), dtype=bool)
    is_train[train_indexes] = True

    train_source = [i for i, j in zip(source, is_train) if j]
    test_source = [i for i, j in zip(source, is_train) if not j]
    train_target = [i for i, j in zip(target, is_train) if j]
    test_target = [i for i, j in zip(target, is_train) if not j]

    train_source, train_target, test_source, test_target = select_debug(
        config, train_source, train_target, test_source, test_target
    )
    # return train_source.tolist(), train_target.tolist(), test_source.tolist(), test_target.tolist()
    return train_source, train_target, test_source, test_target


def shuffle(
    source: Iterable, target: Iterable, rng: np.random.RandomState
) -> Tuple[List[str], List[str]]:
    """
    Shuffle the dataset

    Args:
        source: source texts
        target: target translations
        rng: rng

    Returns:
        Source texts and target translations
    """
    data = list(zip(source, target))
    rng.shuffle(data)
    source, target = zip(*data)  # Tuple, tuple
    return list(source), list(target)


### Loading ###


def load_tokenizers(
    config: TrainConfig,
) -> Tuple[Data.SentencePieceTokenizer, Data.SentencePieceTokenizer]:
    """Load the tokenizers for the input text and the output text"""
    source_tokenizer = Data.SentencePieceTokenizer(Data.ETRUSCAN)
    target_tokenizer = Data.SentencePieceTokenizer(Data.ETRUSCAN)
    source_tokenizer.load(config.source_model)
    target_tokenizer.load(config.target_model)
    return source_tokenizer, target_tokenizer


def load_data(config: TrainConfig) -> Tuple[List[str], List[str]]:
    """
    Load and clean the data

    Args:
        config: parameters

    Returns:
        List of source texts and list of target translations.
    """
    subset = config.subset
    if subset is None:
        subset = "both"

    lang_pair = {config.source_lang, config.target_lang}
    if lang_pair == {"etruscan", "english"}:
        source, target = Data.load_translation_dataset(
            etruscan_fn=clean_etruscan, english_fn=clean_english, subset=subset
        )
    elif lang_pair == {"greek", "english"} or lang_pair == {"latin", "english"}:
        data = pd.read_csv(config.data_path, index_col=0).dropna(
            subset=["source", "target"]
        )
        source = data["source"].to_list()
        target = data["target"].to_list()
    elif config.source_lang == config.target_lang == "english":
        data = pd.read_csv(config.data_path, index_col=0).dropna(
            subset=["source", "target"]
        )
        if "target" in data.columns:
            target = data["target"].to_list()
        else:
            _, target = Data.load_translation_dataset(
                etruscan_fn=clean_etruscan, english_fn=clean_english
            )
        source = target.copy()
    elif config.source_lang == config.target_lang == "etruscan":
        source, _ = Data.load_translation_dataset(
            etruscan_fn=clean_etruscan, english_fn=clean_english
        )
        target = source.copy()
    elif config.source_lang == config.target_lang:
        data = pd.read_csv(config.data_path, index_col=0).dropna(
            subset=["source", "target"]
        )
        source = data["source"].to_list()
        target = source.copy()

    if config.source_lang == "english":
        target, source = source, target
    return source, target


def make_dataloader(
    source: List[str],
    source_chars: List[List[int]],
    source_words: List[List[int]],
    target: List[str],
    target_chars: List[List[int]],
    target_words: List[List[int]],
    config: TrainConfig,
    max_len: int,
) -> DataLoader:
    """
    Create a dataloader for Larth.

    Note: the char and word sequences must be already aligned (i.e. same length)

    Args:
        source: source texts
        source_chars: tokenized source texts (chars)
        source_words: tokenized source texts (words)
        target: target texts
        target_chars: tokenized target texts (chars)
        target_words: tokenized target texts (words)
        config: parameters
        max_len: maximum length allowed for the sequences (longer sequences are truncated)

    Returns:
        Dataloader with the source and target data.
    """
    assert (
        len(source_chars)
        == len(source_words)
        == len(target_chars)
        == len(target_words)
        == len(source)
        == len(target)
    )
    table = pa.Table.from_pydict(
        {
            "source": source,
            "source_chars": source_chars,
            "source_words": source_words,
            "target": target,
            "target_chars": target_chars,
            "target_words": target_words,
        }
    )
    dl = DataLoader(
        table, batch_size=config.batch_size, cached=config.cached, max_len=max_len
    )
    return dl


def get_training_data(
    config: TrainConfig, max_len: int = 1024
) -> Tuple[
    DataLoader, DataLoader, Data.SentencePieceTokenizer, Data.SentencePieceTokenizer
]:
    """
    Get train and test datasets and the tokenizers

    Args:
        config: parameters
        max_len: maximum length allowed for the sequences (longer sequences are truncated)

    Returns:
        Train dataloader, test dataloader, source language tokenizer, target language tokenizer

    Note: the tokenizers can be identical
    """
    rng = np.random.RandomState(config.seed)

    # Tokenizers
    source_tokenizer, target_tokenizer = load_tokenizers(config)

    # Load & clean (Greek/Latin data is already clean)
    if config.dataset_type == "csv":
        source, target = load_data(config)

        # No duplicates, no empty
        source, target = remove_invalid(source, target)

        # Split
        train_source, train_target, test_source, test_target = split(
            source, target, config, rng
        )
    elif config.dataset_type == "tatoeba":
        # Load
        (
            train_source,
            train_target,
            test_source,
            test_target,
            dev_source,
            dev_target,
        ) = Data.load_tatoeba(config.data_path)
        if config.subset == "dev":
            train_source = dev_source.copy()
            train_target = dev_target.copy()
            test_source = dev_source.copy()
            test_target = dev_target.copy()
        if config.debug:
            train_source, train_target, test_source, test_target = select_debug(
                config, train_source, train_target, test_source, test_target
            )

        # HACK
        # Assume that it is either eng-X or X-eng
        if "eng-" in config.data_path and config.source_lang != "english":
            train_source, train_target = train_target, train_source
            test_source, test_target = test_target, test_source

        # Remove invalid
        train_source, train_target = remove_invalid(train_source, train_target)
        test_source, test_target = remove_invalid(test_source, test_target)

        # Clean
        train_source = list(map(clean_tatoeba, train_source))
        test_source = list(map(clean_tatoeba, test_source))

        train_target = list(map(clean_tatoeba, train_target))
        test_target = list(map(clean_tatoeba, test_target))

        # train_source, train_target, test_source, test_target = select_debug(config, train_source, train_target, test_source, test_target)
    else:
        raise ValueError(
            f"Invalid language pair: {config.source_lang}, {config.target_lang}"
        )

    # Data augmentation
    train_source, train_target = augment(train_source, train_target, config, rng)

    # Shuffle in pairs
    train_source, train_target = shuffle(train_source, train_target, rng)
    test_source, test_target = shuffle(test_source, test_target, rng)

    # Tokenization
    assert len(train_source) == len(train_target)
    assert len(test_source) == len(test_target)

    train_source_chars, train_source_words = source_tokenizer.tokenize(
        train_source, align=True, align_mode=config.alignment
    )

    test_source_chars, test_source_words = source_tokenizer.tokenize(
        test_source, align=True, align_mode=config.alignment
    )

    train_target_chars, train_target_words = target_tokenizer.tokenize(
        train_target, align=False
    )

    test_target_chars, test_target_words = target_tokenizer.tokenize(
        test_target, align=False
    )

    # Datasets
    train_dl = make_dataloader(
        train_source,
        train_source_chars,
        train_source_words,
        train_target,
        train_target_chars,
        train_target_words,
        config,
        max_len,
    )
    test_dl = make_dataloader(
        test_source,
        test_source_chars,
        test_source_words,
        test_target,
        test_target_chars,
        test_target_words,
        config,
        max_len,
    )

    # Size and vocab size
    logging.info(f"Train examples ({config.source_lang}): {len(train_source_chars)}")
    logging.info(f"Test examples ({config.target_lang}): {len(test_source_chars)}")
    logging.info(f"Char vocab size (source): {source_tokenizer.vocab_size(False)}")
    logging.info(f"Word vocab size (source): {source_tokenizer.vocab_size()}")
    logging.info(f"Char vocab size (target): {target_tokenizer.vocab_size(False)}")
    logging.info(f"Word vocab size (target): {target_tokenizer.vocab_size()}")

    return train_dl, test_dl, source_tokenizer, target_tokenizer
