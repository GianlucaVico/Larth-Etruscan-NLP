"""
Methods and classes for working on the datasets.

Variables:
    name_columns: columns in the POS dataframe that indicates that the token is a proper name
    categories: columns in the POS with the grammmar categories
"""
import pandas as pd
import re
from ast import literal_eval
from typing import List, Tuple, Callable, Dict

try:
    from . import utils
except ModuleNotFoundError:
    import utils

_dir = __file__.rsplit("/", 1)[0] + "/"

name_columns: List[str] = [
    "city name",
    "place name",
    "name",
    "epithet",
    "theo",
    "cogn",
    "prae",
    "nomen",
]

categories: List[str] = [
    "city name",
    "place name",
    "name",  # Unspecified (?) name
    "epithet",
    "theo",  # Theonomin
    "cogn",  # Cognomen
    "prae",  # Praenomen
    "nomen",  # Nomen
    "nom",  # Nominative
    "acc",  # Accusative
    "masc",  # Masculine
    "fem",  # Feminine
    "nas-part",
    "nasa-part",
    "u-part",
    "θ-impv",  # θ-Imperative
    "θ-part",
    "θas-part",
    "as-part",
    "act",  # Active
    "pass",  # Passive
    "non-past",
    "past",  # Past
    "impv",  # Imperative
    "jussive",
    "necess",
    "inanim",  # Inanimate
    "anim",  # Animate
    "indef",  # Indefinite (pronoun)
    "def",  # Definite (article)
    "deictic particle",
    "enclitic particle",
    "enclitic conj",
    "dem",  # Demonstrative
    "adv",  # Adverb
    "art",  # Article
    "conj",  # Conjunction
    "post",  # Post-position
    "pro",  # Pronoun
    "rel",  # Relative
    "subord",  # Subordinator
    "neg",
    "num",  # Numeral
    # "particle",
    "1st gen",  # Genitive (1st/2nd)
    "2nd gen",
    "1st abl",  # Ablative (1st/2nd)
    "2nd abl",
    "loc",  # Locative
    "1st pert",  # Pertinentive
    "2nd pert",
    "1st pers",  # 1st/2nd/3rd Person/Personal
    "2nd pers",
    "3rd pers",
    "pl",  # Plural
]


def load_pos(path: str = _dir + "ETP_POS.csv") -> pd.DataFrame:
    """
    Load the dictionary as pandas dataframe

    Args:
        path: path to the CSV file

    Returns:
        Dataframe
    """
    return pd.read_csv(
        path,
        index_col=0,
        true_values=["True", "TRUE", "true"],
        false_values=["False", "False", "false"],
        converters={"Translations": literal_eval},
    )


def load_translation_vocab(path: str = _dir + "ETP_POS.csv") -> List[Tuple[str, str]]:
    """
    Load the list of words and translations

    Args:
        path: path to the csv file

    Returns:
        List of pair word-translation
    """
    vocab = load_pos(path)
    pairs = []
    for row in vocab.iloc:
        et = row["Etruscan"]
        for i in row["Translations"]:
            pairs.append((et, i[1]))
    return pairs


def is_proper_name(df: pd.DataFrame, include_abbreviation: bool = False) -> pd.Series:
    """
    Check if the entry is a proper name.

    Args:
        df: POS dataframe
        include_abbreviation: whether to include abbreviated names in the name mask

    Returns:
        Bool mask that selects the proper names
    """
    mask = df[name_columns].apply(pd.Series.any, axis=1)
    if not include_abbreviation:
        mask = mask & df["Abbreviation of"].isna()  # If NA -> not an abbreviation
    return mask


def load_translation_dataset(
    path: str = _dir + "Etruscan.csv",
    subset: str = "both",
    to_latin: bool = True,
    etruscan_fn: Callable[[str], str] | None = None,
    english_fn: Callable[[str], str] | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Load the data for the translation task directly.

    Args:
        path: path to the csv file
        subset: whether to load only CIEP, only ETP or both (values in ["both", "etp", "ciep"])
        to_latin: whether to conver the Etruscan text to Latin
        etruscan_fn: function to apply to the Etruscan text
        english_fn: function to apply to the English text

    Returns:
        List of Etruscan texts and list of English texts
    """
    subset = subset.lower().strip()
    if subset not in ["both", "etp", "ciep"]:
        raise ValueError("Subset must be 'both', 'ciep' or 'etp'")

    data = pd.read_csv(path, index_col=0)
    data = data.dropna(subset=["Translation"]).reset_index()
    data["Translation"] = data["Translation"].apply(str.lower)

    if to_latin:
        data["Etruscan"] = data["Etruscan"].apply(
            lambda x: utils.replace(x, utils.to_latin)
        )
        data["Translation"] = data["Translation"].apply(
            lambda x: utils.replace(x, utils.to_latin)
        )

    if etruscan_fn is not None:
        data["Etruscan"] = data["Etruscan"].apply(etruscan_fn)

    if english_fn is not None:
        data["Translation"] = data["Translation"].apply(english_fn)

    if subset == "ciep":
        data = data.dropna(subset=["key"])
    elif subset == "etp":
        data = data[data["key"].isna()]

    return data["Etruscan"].to_list(), data["Translation"].to_list()


def load_lm_dataset(
    path: str = _dir + "Etruscan.csv",
    subset: str = "both",
    to_latin: bool = True,
    etruscan_fn: Callable[[str], str] | None = None,
) -> List[str]:
    """
    Load the data for the lnaguage modelling task directly.

    Args:
        path: path to the csv file
        subset: whether to load only CIEP, only ETP or both (values in ["both", "etp", "ciep"])
        to_latin: whether to conver the Etruscan text to Latin
        etruscan_fn: function to apply to the Etruscan text
    Returns:
        List of Etruscan texts
    """
    subset = subset.lower().strip()
    if subset not in ["both", "etp", "ciep"]:
        raise ValueError("Subset must be 'both', 'ciep' or 'etp'")

    data = pd.read_csv(path, index_col=0)

    if to_latin:
        data["Etruscan"] = data["Etruscan"].apply(
            lambda x: utils.replace(x, utils.to_latin)
        )

    if etruscan_fn is not None:
        data["Etruscan"] = data["Etruscan"].apply(etruscan_fn)

    if subset == "ciep":
        data = data.dropna(subset=["key"])
    elif subset == "etp":
        data = data[data["key"].isna()]

    return data["Etruscan"].to_list()


def load_suffixes(
    suffix_list: str | None,
    vocab_csv: str | None,
    alphabet_map: Dict[str, str] | None,
) -> Tuple[List[str], List[str]]:
    """
    Generate a list of terminal (no other suffixes after them) and
    non-terminal (there can be other suffixes after them)suffixes.

    Args:
        suffix_list: test file with a suffix per line (-aaa-: non-terminal; -aaa: terminal)
        vocab_csv: csv file for the dictionary
        alphabet_map: dictionary for replacing characters (e.g. for text normalisation or cleaning)

    Returns:
        List of non-terminal suffixes and list of terminal suffixes
    """

    non_terminal_suffixes = []
    terminal_suffixes = []

    # Read suffix list
    if suffix_list is not None:
        with open(suffix_list) as f:
            lines = f.readlines()
        if alphabet_map is not None:
            lines = [
                utils.replace(i.strip(), alphabet_map) for i in lines if i is not None
            ]
        else:
            lines = [i.strip() for i in lines if i is not None]
        lines = [i for i in lines if i != ""]

        for i in lines:
            if i.endswith("-"):
                # e.g. -i- -> add i
                non_terminal_suffixes.append(i[1:-1])
            else:
                # e.g. -i -> add i
                terminal_suffixes.append(i[1:])

    # Read POS file
    if vocab_csv is not None:
        vocab = load_pos(vocab_csv)
        vocab = vocab[vocab["Is suffix"]]["Etruscan"].dropna()
        vocab = (
            vocab.apply(str.strip)
            .apply(str.lower)
            .apply(lambda x: re.sub("[\[\]\(\)<>]", "", x))
        )
        if alphabet_map is not None:
            vocab = vocab.apply(lambda x: utils.replace(x, alphabet_map))
        terminal_suffixes.extend(vocab.to_list())

    # Remove duplicated
    terminal_suffixes = list(set(terminal_suffixes))
    non_terminal_suffixes = list(set(non_terminal_suffixes))

    # Sort to avoid partially removing a suffix -> longer first
    sort_key = lambda x: len(x)
    terminal_suffixes = sorted(terminal_suffixes, key=sort_key, reverse=True)
    non_terminal_suffixes = sorted(non_terminal_suffixes, key=sort_key, reverse=True)
    return non_terminal_suffixes, terminal_suffixes


def load_tatoeba(
    path: str,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Args:
        path: directory containing the dataset (train.src.gz, train.trg.gz, etc...)

    Retuns:
        List of string for train source, train target, test source, test target, dev source, dev target
    """
    if not path.endswith("/"):
        path = path + "/"

    files = ["train.src", "train.trg", "test.src", "test.trg", "dev.src", "dev.trg"]

    splits = [None] * len(files)

    for i, file in enumerate(files):
        with open(path + file) as f:
            splits[i] = f.readlines()
    splits = [[j.lower().strip() for j in i] for i in splits]
    return splits
