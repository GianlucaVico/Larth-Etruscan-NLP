"""Data module: function to read the dataset and clean it"""
from . import (
    tokenizers,
    data,
    augmentation_base,
    augmentation_bi,
    augmentation_mono,
    pos,
    utils,
)

from .data import (
    load_pos,
    is_proper_name,
    load_translation_dataset,
    load_tatoeba,
    load_lm_dataset,
)
from .augmentation_base import create_index, index_df_to_map
from .augmentation_mono import mark_text_mono, generate_mono, generate_etruscan
from .augmentation_bi import mark_text_bi, generate_bi, generate_pairs
from .pos import (
    only_alpha,
    make_pos_train_set,
    make_category_train_set,
    simple_tokenizer,
    tag,
    get_categories,
    category_description,
)

from .tokenizers import (
    TokenizerType,
    BaseTokenizer,
    BlankspaceTokenizer,
    SuffixTokenizer,
    SentencePieceTokenizer,
    ETRUSCAN,
    ENGLISH,
)

from .utils import (
    greek_to_latin,
    to_latin,
    to_extended_latin,
    replace,
    parenthesis_re_no_space,
    parenthesis_re,
    curly_brakets_re,
    not_alphanum_re,
    date_re,
    brakets_re,
    low_brakets_re,
    tags,
    T_re,
    C_re,
    A_re,
)

_dir: str = __file__.rsplit("/", 1)[0] + "/"
