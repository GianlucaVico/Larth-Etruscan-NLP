"""
Functions for data augmentation on the Etruscan texts (monolingual)
"""
from typing import Tuple, List, Dict
import pandas as pd
import re
from collections import deque
import random
from tqdm import tqdm

try:
    from .augmentation_base import *
except ImportError:
    from augmentation_base import *

tqdm.pandas()


def mark_text_mono(
    text: str, index_df: pd.DataFrame | Dict[str, int], fmt: str = "§{}§"
) -> str:
    """
    Replace proper names with an index.

    Args:
        text: text to process
        index_df: index dataframe
        fmt: string to format the indexes (e.g., "§0§", "§10§")
    Return:
        Text with index instead of the proper names.
    """
    if isinstance(index_df, pd.DataFrame):
        index_df = index_df_to_map(index_df, "et", True)

    for name, index in index_df.items():
        r = re.compile(rf"\b{name}\b")
        text = r.sub(fmt.format(index), text)
    return text


def generate_mono(
    text: str,
    index_df: pd.DataFrame | Dict[int, List[str]],
    fmt: str = "§{}§",
    index_threshold: Tuple[int, int] = (8, 20),
    max_replacements: Tuple[int, int, int] = (3, 2, 1),
    rng: random.Random = None,
) -> List[str]:
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
        rng = random.Random(0)

    if isinstance(index_df, pd.DataFrame):
        index_df = index_df_to_map(index_df, "et", False)
    # RE to find the marks
    mark = re.compile(fmt.format(r"(?P<index>[0-9]+)"))

    # Results, list of strings
    out = []

    # Strore string that could have marks in it
    q = deque()
    q.append(text)

    n = len(mark.findall(text))
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
        match_ = mark.search(t)

        if match_ is None:  # All substitution are done
            out.append(t)
        else:  # Still some mark to replace
            index = int(match_.group("index"))
            this_mark = re.compile(match_.group())  # e.g., §0§ instead of §.*§
            # candidates = index_df[index_df["Index"] == index][col].to_list()
            candidates = index_df[index]

            # Too many raplacements: select few
            if len(candidates) > threshold:
                # candidates = rng.choice(candidates, threshold)
                candidates = rng.sample(candidates, k=threshold)

            for c in candidates:
                q.append(this_mark.sub(c, t, 1))  # Replace only the first match

    return out


def generate_etruscan(
    docs: pd.DataFrame,
    index_df: pd.DataFrame,
    index_threshold: Tuple[int, int] = (8, 20),
    max_replacements: Tuple[int, int, int] = (3, 2, 1),
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Generate new Etruscan texts.

    Args:
        docs: dataframe with Etruscan texts. Column: "Etruscan"
        index_df: index dataframe
        index_threshold: use a different number of replacements based on the number of indexes
        max_replacements: up to this number of replacements for each index
    Return:
        Tuple with generated texts and marked texts
    """
    rng = random.Random(seed)
    out = [None] * len(docs)  # List of lists

    name_to_index = index_df_to_map(index_df, "et", True)
    index_to_name = index_df_to_map(index_df, "et", False)

    # Add the indexes
    marked = (
        docs["Etruscan"]
        .progress_apply(lambda x: mark_text_mono(x, name_to_index))
        .to_list()
    )

    # mark = re.compile(r"§(?P<index>[0-9]+)§")

    # Replace the indexes
    for i, j in enumerate(tqdm(marked)):
        out[i] = generate_mono(
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
