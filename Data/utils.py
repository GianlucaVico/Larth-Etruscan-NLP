"""
RE:
    - parenthesis_re: text in parenthesis (like this one) but not l(i)k(e) t(his)
    - brakets_re
    - date_re: dates likes 500 or 500 - 400
    - T_re: lines with Etruscan text in CIEP
    - C_re: lines with reconstructed Etruscan text in CIEP
    - A_re: lines with translations in CIEP

Dictionaries:
    - to_latin: replace characters with one or more characters from the Latin alphabet. This should be the convention used in CIEP.
    - to_extended_lating: replace multiple characters with a single character from the Greek or Latin alphabet. Used to clean earlier 
        versions of ETP.csv (ç is faster to type than ς with an Italian keyboard)
Lists:
    - tags: list of tags describing Etruscan words
"""
import re
from typing import Dict, List
import numpy as np

# parenthesis_re = re.compile("(^| +)\([a-zA-Z]+\)($| +)")
parenthesis_re = re.compile("(^| |[^a-zA-Z])\([^)]*\)( |[^a-zA-Z]|$)")
parenthesis_re_no_space = re.compile("\([^)]*\)")
brakets_re = re.compile("\[[^]]*\]")
curly_brakets_re = re.compile("\{[^}]*\}")
low_brakets_re = re.compile("\<[^>]*\>")

date_re = re.compile("[0-9]+( ?- ?[0-9]+)?")

Inscription_re = re.compile("(INSCRIPTIO NVMERO \d*.)|(^T\d)|(^C\d)|(^A\d)")
T_re = re.compile("T *[0-9]+[a-zA-Z]*(\|[T][0-9]+[a-zA-Z]*)?")
C_re = re.compile("C *[0-9]+[a-zA-Z]*(\|[C][0-9]+[a-zA-Z]*)?")
A_re = re.compile("A *[0-9]+[a-zA-Z]*(|[A][0-9]+[a-zA-Z]*)?")

# not_alphanum_re = re.compile("[\W_]")
not_alphanum_re = re.compile("[^a-zA-Z0-9_ ]")

to_latin: Dict[str, str] = {
    "ς'": "sh",
    "ς": "s",
    "θ": "th",
    "σ'": "sh",
    "σ": "s",
    "χ": "kh",
    "φ": "ph",
    ":": " ",
    "|": "",
    ".": "",
    "\t": " ",
    "\n": " ",
    "/": "",
    "\\": "",
    "š": "sh",
    "‘": "",
    "’": "",
    "'": "",
    "ś": "sh",
    "ê": "e",
    "•": " ",
    # "[": "",
    # "]": "",
    "(": "",
    ")": "",
    "{": "",
    "}": "",
    ";": "",
    # "-": "", # Remove missing letters for now
    "–": "-",
    "·": "",
    "s^+": "s",
    "|": " ",
    "s'": "sh",
    "s*": "s",
    "s^x": "sh",
    "0": "th",
    "‘": "'",
    "’": "'",
    "ç": "s",
    "x*": "kh",
    "f*": "ph",
    "e'": "e",
}

to_extended_latin: Dict[str, str] = {
    "s'": "ś",
    "s*": "σ",
    "s^x": "š",
    "0": "θ",
    "‘": "'",
    "’": "'",
    "ç": "ς",
    "x*": "χ",
    "f*": "φ",
    # "s^+":
    "e'": "ê",
    "|": " | ",
    # ":" : " : ",
    "  ": " ",
}


def replace(s: str | None, d: Dict[str, str]) -> str | None:
    """
    Replace substrings in the given string

    Args:
        s: string
        d: dictionary with the replacements
    Returns:
        New string (if the input string is not None)
    """
    if s is not None and s is not np.nan:
        for k, v in d.items():
            s = s.replace(k, v)
    return s


# DEPRECATED: moved to Data
tags: List[str] = [
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

# Simple Greek transliteration (inaccurate)
greek_to_latin = {
    "α": "a",
    "ά": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "έ": "e",
    "ε": "e",
    "ζ": "z",
    "η": "h",
    "ή": "h",
    "θ": "u",
    "ι": "i",
    "ί": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "j",
    "ο": "o",
    "ό": "o",
    "π": "p",
    "ρ": "r",
    "ς": "s",
    "σ": "s",
    "τ": "t",
    "ύ": "y",
    "υ": "y",
    "φ": "f",
    "χ": "x",
    "ψ": "c",
    "ω": "v",
    "ώ": "v",
}

greek_to_latin = {ord(k): v for k, v in greek_to_latin.items()}  # For str.translate

others = {ord("æ"): "ae", ord("œ"): "oe"}
