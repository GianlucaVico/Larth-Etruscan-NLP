"""
Functions for data augmentation on the Etruscan and English texts (bilingual)
"""
from typing import Tuple, List, Dict
import pandas as pd
import re
from collections import deque
import numpy as np
import random
from tqdm import tqdm

try:
    from .augmentation_base import *
except ImportError:
    from augmentation_base import *

tqdm.pandas()


def mark_text_bi(
    pair: Tuple[str, str],
    pair_index: Dict[str, int] | None = None,
    fmt: str = "§{}§",
) -> str:
    """
    Replace proper names with an index.

    Args:
        text: text to process
        pair_index_df: index dataframe
        fmt: string to format the indexes (e.g., "§0§", "§10§")
    Return:
        Text with index instead of the proper names.
    """
    if isinstance(pair_index, pd.DataFrame):
        pair_index = index_df_to_map(pair_index, "bi", name_to_index=True)

    for (name_et, name_eng), index in pair_index.items():
        if (
            name_et is not np.nan
            and name_eng is not np.nan
            and len(name_et) != 0
            and len(name_eng) != 0
        ):
            r_et = re.compile(rf"\b{name_et}\b")
            r_eng = re.compile(rf"\b{name_eng}\b")

            if len(r_et.findall(pair[0])) == len(r_eng.findall(pair[1])) != 0:
                pair = (
                    r_et.sub(fmt.format(index), pair[0]),
                    r_eng.sub(fmt.format(index), pair[1]),
                )
    return pair


def generate_bi(
    pair: Tuple[str, str],
    pair_index: pd.DataFrame | Dict[str, int],
    fmt: str = "§{}§",
    index_threshold: Tuple[int, int] = (8, 20),
    max_replacements: Tuple[int, int, int] = (3, 2, 1),
    rng: random.Random = None,
) -> List[Tuple[str, str]]:
    """
    Replace the index with all the compatible proper names.

    Args:
        text: text to process
        index_df: index dataframe
        etruscan: wheter it is an Etruscan text. English otherwise
        fmt: string to format the indexes (e.g., "§0§", "§10§")
        index_threshold: use a different number of replacements based on the number of indexes in the inscription
        max_replacements: up to this number of replacements for each index in the inscrion (based on the index count and index_threshold)
        rng: random number generator used to select the replacements
    Return:
        List of text with proper names instead of indexes.

    Note: it might generate duplicated entries
    """
    if rng is None:
        rng = np.random.RandomState(0)

    if isinstance(pair_index, pd.DataFrame):
        pair_index = index_df_to_map(pair_index, "bi", False)
    # RE to find the marks
    mark = re.compile(fmt.format(r"(?P<index>[0-9]+)"))

    # Results, list of strings
    out = []

    # Strore string that could have marks in it
    q = deque()
    q.append(pair)

    n = len(mark.findall(pair[0]))
    # print(n)

    if n > index_threshold[1]:
        threshold = max_replacements[2]
    elif n > index_threshold[0]:
        threshold = max_replacements[1]
    else:
        threshold = max_replacements[0]

    while len(q) != 0:
        # Get next string
        t = q.popleft()
        match_et = mark.search(t[0])
        # match_eng = mark.search(t[1])

        if match_et is None:  # All substitution are done
            out.append(t)
        else:  # Still some mark to replace
            index = int(match_et.group("index"))
            this_mark = re.compile(match_et.group())  # e.g., §0§ instead of §.*§
            # candidates = index_df[index_df["Index"] == index][col].to_list()
            candidates = pair_index[index]

            # Too many raplacements: select few
            if len(candidates) > threshold:
                # candidates = rng.choice(candidates, threshold, replace=False)
                candidates = rng.sample(candidates, k=threshold)

            for c in candidates:
                q.append(
                    (this_mark.sub(c[0], t[0], 1), this_mark.sub(c[1], t[1], 1))
                )  # Replace only the first match

    return out


def generate_pairs(
    docs: pd.DataFrame,
    index_df: pd.DataFrame,
    index_threshold: Tuple[int, int] = (8, 20),
    max_replacements: Tuple[int, int, int] = (3, 2, 1),
    seed: int = 0,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generate new Etruscan texts.

    Args:
        docs: dataframe with Etruscan texts. Column: "Etruscan", "Translations"
        index_df: index dataframe
        index_threshold: use a different number of replacements based on the number of indexes in the inscription
        max_replacements: up to this number of replacements for each index in the inscrion (based on the index count and index_threshold)
    Return:
        Tuple with generated texts and marked texts
    """
    docs = docs.dropna(subset=["Etruscan", "Translation"])
    rng = random.Random(seed)
    out = [None] * len(docs)  # List of lists

    name_to_index = index_df_to_map(index_df, "bi", True)
    index_to_name = index_df_to_map(index_df, "bi", False)

    # Add the indexes
    marked = docs.progress_apply(
        lambda x: mark_text_bi((x["Etruscan"], x["Translation"]), name_to_index), axis=1
    ).to_list()

    # mark = re.compile(r"§(?P<index>[0-9]+)§")

    # Replace the indexes
    for i, j in enumerate(tqdm(marked)):
        out[i] = generate_bi(
            j,
            index_to_name,
            index_threshold=index_threshold,
            max_replacements=max_replacements,
            rng=rng,
        )

    # Flatten the output
    out = [j for i in out for j in i]
    out = list(set(out))
    return out, marked
