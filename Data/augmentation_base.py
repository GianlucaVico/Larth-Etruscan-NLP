"""
Base functions for data augmentation.

See `augmentation_bi.py` for data augmentation for bilingual data and 
`augmentation_mono.py` for monolingual data
"""
from typing import Tuple, List, Dict
import pandas as pd
from collections import defaultdict
import numpy as np

try:
    from .data import categories, is_proper_name
except ImportError:
    from data import categories, is_proper_name


def compute_index(df: pd.DataFrame) -> Tuple[List[int], Dict[Tuple[bool], int]]:
    """
    Compute the indexes for the proper names. Names with the same characteristics
    have the same index.

    Args:
        df: dataframe with the category columns of the proper names (i.e., exclude "Translation", "POS", etc..., keep "nom", "acc", etc...)
    Return:
        Tuple with list of indexes and dictionary with tuple describing the name and the index.
    """
    indexes = []
    map_ = {}
    current_index = -1
    for row in df.iloc:
        tmp = tuple((row >= 0.5).to_list())
        candidate = map_.get(tmp)
        if candidate is None:  # New item
            current_index += 1
            indexes.append(current_index)
            map_[tmp] = current_index
        else:  # Not new
            indexes.append(candidate)

    return indexes, map_


def expand_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the translations in case an inscription as multple: a single translation for each row.

    Args:
        df: dataframe with Etruscan, Translations and Index
    Returns:
        Dataframe with expanded translations
    """
    tmp = []
    for row in df.iloc:
        if len(row["Translations"]) == 0:
            tmp.append((row["Etruscan"], np.nan, row["Index"]))
        else:
            for t in row["Translations"]:
                tmp.append((row["Etruscan"], t[1], row["Index"]))
    return pd.DataFrame.from_records(tmp, columns=["Etruscan", "Translations", "Index"])


def create_index(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Dict[Tuple[bool], int]]:
    """
    Create the index dataframe.

    Args:
        df: POS dataframe
    Returns:
        Index dataframe, name mask, index map
    """
    name_mask = is_proper_name(df)
    indexes, map_ = compute_index(df[name_mask][categories])

    index_df = pd.DataFrame(
        {
            "Etruscan": df[name_mask]["Etruscan"],
            "Translations": df[name_mask]["Translations"],
            "Index": indexes,
        }
    ).reset_index(drop=True)
    index_df = expand_index(index_df)
    return index_df, name_mask, map_


def index_df_to_map(
    index_df: pd.DataFrame, lang: str = "et", name_to_index: bool = True
) -> (
    Dict[str, int]
    | Dict[Tuple[str, str], int]
    | Dict[int, List[str]]
    | Dict[int, List[Tuple[str, str]]]
):
    """
    Convert the index dataframe to a dictionary for efficiency

    Args:
        index_df: index dataframe
        lang: language of the map (values in ["et", "eng", "bi"], "bi" for bilingual)
        name_to_index: if True the output maps a name to an index. Otherwise, it maps an index to a list of names
    Returns:
        Dictionary
    """
    if lang == "et":
        col = "Etruscan"
    elif lang == "eng":
        col = "Translations"
    elif lang == "bi":
        col = ["Etruscan", "Translations"]
    index_df = index_df.dropna()
    if name_to_index:
        if lang == "bi":
            # Dict: (et name, eng name): index
            return dict(
                zip(
                    list(index_df[col].itertuples(index=False, name=None)),
                    index_df["Index"],
                )
            )
        else:
            # Dict: name: index
            return dict(zip(index_df[col].to_list(), index_df["Index"]))
    else:
        d = defaultdict(list)
        if lang == "bi":
            for row in index_df.iloc:
                d[row["Index"]].append(tuple(row[col]))
        else:
            for row in index_df.iloc:
                d[row["Index"]].append(row[col])
        return d
