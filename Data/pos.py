"""
Methods to annotate POS and grammatical categories.
"""
import re
import pandas as pd
from typing import List, Tuple, Dict, Iterable
import nltk

try:
    from .data import categories
except ImportError:
    from data import categories


def only_alpha(t: str) -> str:
    """
    Args:
        t: some text

    Returns:
        Remove all characters but Latin letters and space from the input
    """
    return re.sub(r"[^a-zA-Z ]", "", t)


def make_pos_train_set(vocab: pd.DataFrame) -> List[List[Tuple[str, str]]]:
    """
    Use the vocabulary/POS dataframe to create a train set for the POS tagger

    Args:
        vocab: dataframe with the columns "Etruscan" and "TAG"

    Returns:
        List of training examples. Each item is a list of annotated tokens (only one token).
        The annotation is `(token, tag)`.
    """
    words = vocab["Etruscan"].apply(only_alpha)
    tags = vocab["TAG"]
    return [[(i, j)] for i, j in zip(words, tags)]


def simple_tokenizer(x: str) -> List:
    """
    Note: use `tokenizers.BlankSpaceTokenizer` instead

    White space tokenizer. It uses ":",  "•" and " " to split the tokens.

    Args:
        x: some text

    Returns:
        List of tokens
    """
    tmp = [i.strip() for i in re.split(r"[:• ]", x.lower())]
    return [i for i in tmp if len(i) != 0]


def tag(tagger: nltk.tag.TaggerI, docs: pd.DataFrame) -> List[List[Tuple[str, str]]]:
    """
    Tag a list of Etruscan documents

    Args:
        tagger: any nltk tagger
        docs: dataframe with the column Etruscan

    Returns:
        List of annotated sentences. Each sentens is a list of pairs `(token, tag)`.
    """
    toks = docs["Etruscan"].apply(simple_tokenizer)
    return [tagger.tag(i) for i in toks]


def get_categories(multi_case: bool = True) -> List[str]:
    """
    Get the list of grammatical categories.
    Genitive, ablative and pertinetive can have two case.

    Args:
        multi_case: whether to return 1st and 2nd cases or not (e.g., "1st gen" and "2nd gen" or "gen")

    Returns:
        List of grammatical categories
    """
    cols = categories.copy()
    if not multi_case:
        cases = ["gen", "abl", "pert"]
        for i in ["1st", "2nd"]:
            for j in cases:
                cols.remove(f"{i} {j}")
        cols += cases
    return cols


def make_category_train_set(
    vocab: pd.DataFrame, multi_case: bool = True
) -> List[List[Tuple[str, Tuple]]]:
    """
    Use the vocabulary/POS dataframe to create a train set for the tagger for the grammatical categories

    Args:
        vocab: dataframe with the columns "Etruscan" and "TAG"
        multi_case: whether to return 1st and 2nd cases or not (e.g., "1st gen" and "2nd gen" or "gen")

    Returns:
        List of training examples. Each item is a list of annotated tokens (only one token).
        The annotation is `(token, tuple of categories)`.
    """
    cols = get_categories(multi_case)
    words = vocab["Etruscan"].apply(only_alpha)
    tags = vocab[cols].itertuples(index=False, name=None)
    return [[(i, j)] for i, j in zip(words, tags)]


def category_description(
    cat: Iterable, to_dict: bool = False
) -> Dict[str, float | bool] | str:
    """
    Get the description of a word given the grammatical features

    Args:
        cat: values of the grammtical features
        to_dict: whether to return a multi-line string or a dictionary

    Returns:
        Human readable string or dictionary describing the grammatical features.
    """
    if len(cat) == 54:
        names = get_categories(multi_case=True)
    elif len(cat) == 51:
        names = get_categories(multi_case=False)
    else:
        raise Exception("Invalid category length")

    d = dict(zip(names, cat))
    if to_dict:
        return d
    else:
        lines = [f"{i}: {j}" for i, j in d.items()]
        return "\n".join(lines)
