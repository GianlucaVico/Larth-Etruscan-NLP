"""
Load the dataset and the tokenizers
"""
import logging
import re
from typing import List, Tuple
import numpy as np
from train_utils import TrainConfig, DataLoader
import pyarrow as pa
import sys

sys.path.append("../")
sys.path.append("../..")
import Data

remove_chars = re.compile(r"[126\[\],<>]")
space_norm = re.compile(r" +")


def clean_etruscan(x: str) -> str:
    """
    Args:
        x: Etruscan string

    Returns:
        Clean Etruscan string
    """
    x = x.lower()
    x = remove_chars.sub(" ", x)
    x = space_norm.sub(" ", x)
    return x.strip()


def load_tokenizer(config: TrainConfig) -> Data.SentencePieceTokenizer:
    """
    Load the word tokenizer for the configuration

    Args:
        config: train configuration

    Returns:
        SentencePiece tokenizer
    """
    tokenizer = Data.SentencePieceTokenizer(Data.ETRUSCAN)
    tokenizer.load(config.tokenizer)
    return tokenizer


def load_data(config: TrainConfig) -> List[str]:
    """
    Load the Etruscan inscriptions from the configuration

    Args:
        config: train configuration

    Returns:
        List of texts
    """
    subset = config.subset
    if subset is None:
        subset = "both"

    if config.lang == "etruscan":
        texts = Data.load_lm_dataset(
            config.data_path, subset, etruscan_fn=clean_etruscan
        )
    else:
        raise NotImplementedError("LM for other languages not supported")
    return texts


def split(
    texts: List[str],
    config: TrainConfig,
    rng: np.random.RandomState,
) -> Tuple[List[str], List[str]]:
    """
    Split the dataset in train and test

    Args:
        texts: list of examples
        config: train configuration
        rng: random number generator

    Returns:
        List of train examples and list of test examples
    """
    # Select the train samples
    train_indexes = rng.choice(
        len(texts), int(np.ceil(len(texts) * config.train_size)), replace=False
    )

    is_train = np.zeros(len(texts), dtype=bool)
    is_train[train_indexes] = True

    train = [i for i, j in zip(texts, is_train) if j]
    test = [i for i, j in zip(texts, is_train) if not j]

    if config.debug:
        train = train[: min(config.batch_size * 2, len(train))]
        test = test[: min(config.batch_size * 2, len(test))]
    return train, test


def shuffle(texts: List[str], rng: np.random.RandomState) -> List[str]:
    """
    Shuffle the data

    Args:
        texts: list of examples
        rng:

    Returns:
        New list of examples
    """
    rng.shuffle(texts)
    return list(texts)


def make_dataloader(
    texts: List[str], toks: List[List[int]], config: TrainConfig, max_len: int
) -> DataLoader:
    """
    Create the dataloader with the given data.

    Args:
        texts: data
        toks: tokenized data
        config: train configuration
        max_len: maximum sequence length

    Args:
        Dataloader
    """
    assert len(texts) == len(toks)

    table = pa.Table.from_pydict({"source": texts, "source_words": toks})
    dl = DataLoader(table, config.batch_size, config.cached, max_len)
    return dl


def get_training_data(
    config: TrainConfig, max_len: int = 1024
) -> Tuple[DataLoader, DataLoader, Data.SentencePieceTokenizer]:
    """
    Load train and test dataloader and the tokenizer

    Args:
        config: train configuration
        max_len: maximum sequence length

    Returns:
        Train dataloader, test dataloader and tokenizer
    """
    rng = np.random.RandomState(config.seed)

    tokenizer = load_tokenizer(config)

    if config.dataset_type == "csv":
        texts = load_data(config)
        texts = [i for i in texts if len(i.strip()) >= config.min_len]

        # Split
        train, test = split(texts, config, rng)
    else:
        raise NotImplementedError("Other dataset types are not supported")

    # Augmentation
    # TODO

    # Shuffle
    train = shuffle(train, rng)
    test = shuffle(test, rng)

    # Tokenize (words)
    _, train_toks = tokenizer.tokenize(train, False)
    _, test_toks = tokenizer.tokenize(test, False)

    train_dl = make_dataloader(train, train_toks, config, max_len)
    test_dl = make_dataloader(test, test_toks, config, max_len)

    logging.info(f"Train examples: {len(train)}")
    logging.info(f"Test examples: {len(test)}")
    logging.info(f"Word vocab size: {tokenizer.vocab_size()}")

    return train_dl, test_dl, tokenizer
