"""
Module with tokenizers.
"""
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Dict, Tuple, Self
from nltk.tokenize import WhitespaceTokenizer
import re

try:
    from .data import load_suffixes
    from .utils import to_latin
except ImportError:
    from data import load_suffixes
    from utils import to_latin

from itertools import chain

import sentencepiece as spm

ETRUSCAN = list(" abcdefghijklmnopqrstuvwxyz-")  # - is the unk char
ENGLISH = list(" abcdefghijklmnopqrstuvwxyz")

_dir = __file__.rsplit("/", 1)[0] + "/"


class BaseTokenizer(ABC):
    """
    Base class for the tokenizers
    """

    def __call__(self, x: str, *args) -> List[str]:
        """
        Tokenize the input sentence

        Args:
            x: sentence

        Returns:
            List of tokens
        """
        return self.tokenize(x, *args)

    @abstractmethod
    def tokenize(self, x: str, *args) -> List[str]:
        """
        Method kept for compatibility with nltk

        Tokenize the input sentence

        Args:
            x: sentence

        Returns:
            List of tokens
        """
        pass

    def detokenize(self, x: Iterable, *args) -> str:
        """
        Generate text from tokens

        Args:
            x: list of tokens

        Returns:
            Text
        """
        return " ".join(x).strip()

    def vocab_size(self, *args) -> int | None:
        """
        Returns:
            Size the vobaulary if the tokenizers uses one
        """
        return None


TokenizerType = Callable[[str], Iterable[str]] | BaseTokenizer


class BlankspaceTokenizer(BaseTokenizer):
    def __init__(
        self, lower: bool = True, additional_white_spaces: str = ":• "
    ) -> None:
        """
        Wrapper for the nltk WhitespaceTokenizer

        Args:
            lower: if to convert to lower case
            additional_white_space: charasters interpreted as white space
        """
        self._wt: WhitespaceTokenizer = WhitespaceTokenizer()
        self._low: bool = lower
        self._to_white = re.compile(rf"[{additional_white_spaces}]")
        self._to_remove = re.compile("[\<\(\[\{\}\]\)\>]")

    def tokenize(self, x: str) -> List[str]:
        """
        Tokenize the input sentence

        Args:
            x: sentence

        Returns:
            List of tokens
        """
        if self._low:
            x = x.lower()
        x = self._to_white.sub(" ", x)
        x = self._to_remove.sub("", x)
        return self._wt.tokenize(x)

    # def detokenize(self, x: Iterable) -> str:
    #     return " ".join(x).strip()


class SuffixTokenizer(BlankspaceTokenizer):
    def __init__(
        self,
        suffix_list: str | None = _dir + "ETPSuff.txt",
        vocab_csv: str | None = _dir + "ETP_POS.csv",
        alphabet_map: Dict[str, str] | None = to_latin,
    ):
        """
        BlankspaceTokenizer that also separates the suffixes from the root.

        Args:
            suffix_list: list of suffixes. One suffix per line. The suffixes are indicated like "-suffix" of "-suffix-"
            vocab_csv: csv with the vocabulary. It also lists suffixes
            alphabet_map: dictionary to map to the desired alphabet
        """
        super().__init__()
        self._non_terminal_suffixes, self._terminal_suffixes = load_suffixes(
            suffix_list, vocab_csv, alphabet_map
        )

    def tokenize(self, x: str) -> List[str]:
        """
        Tokenize the sentence

        Args:
            x: sentence
        Returns:
            List of tokens
        """

        # Tokenize
        tokens = super().tokenize(x)
        # Split suffixes
        tokens = [self._split(i) for i in tokens]
        # Flatten
        tokens = list(chain.from_iterable(tokens))
        return tokens

    def detokenize(self, x: Iterable) -> str:
        tmp = [("", i) if self._is_suffix(i) else (" ", i) for i in x]
        tmp = chain.from_iterable(tmp)
        return "".join(tmp).strip()

    def _is_suffix(self, tok: str) -> bool:
        """
        Returns:
            if the token is a suffix
        """
        return (tok in self._non_terminal_suffixes) or (tok in self._terminal_suffixes)

    def _split(self, x: str) -> List[str]:
        """
        Args:
            x: token

        Returns:
            Split the token. Return [token] or [token, suff1. , ...]
        """
        suffixes = []
        for s in self._terminal_suffixes:
            if x.endswith(s):
                x = x.removesuffix(s)
                suffixes.append(s)
                break
        for s in self._non_terminal_suffixes:
            if x.endswith(s):
                x = x.removesuffix(s)
                suffixes.append(s)
        # Use the correct order
        suffixes = suffixes[::-1]
        return [x] + suffixes


class NNTOkenizer(BaseTokenizer):
    """
    NOT USED

    Tokenizer for the NN.

    It takes another tokenizer to split a string.
    """

    def __init__(
        self,
        tokenizer: TokenizerType,
        words: List[str],
        alphabet: List[str] = ETRUSCAN,
        suffix_list: str | None = _dir + "ETPSuff.txt",
        vocab_csv: str | None = _dir + "ETP_POS.csv",
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._alphabet = alphabet

        non_term_suff, term_suff = load_suffixes(suffix_list, vocab_csv, to_latin)
        self._w2i = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
        }
        for i in chain(non_term_suff, term_suff, words):
            # do not add duplicates
            if i not in self._w2i:
                self._w2i[i] = len(self._w2i)
        self._i2w = {v: k for k, v in self._w2i.items}

        self._c2i = dict(zip(alphabet, range(len(alphabet))))
        self._i2c = dict(zip(range(len(alphabet), alphabet)))

        self._clean_re = re.compile(rf"[^{''.join(alphabet)}]")

    @property
    def pad(self):
        return self._w2i["<pad>"]

    @property
    def bos(self):
        return self._w2i["<s>"]

    @property
    def eos(self):
        return self._w2i["</s>"]

    @property
    def unk(self):
        return self._w2i["<unk>"]

    def chars_to_ids(self, x: str) -> List[int]:
        return [self._c2i[i] for i in x]

    def ids_to_chars(self, x: Iterable[int]) -> str:
        return "".join([self._i2c[i] for i in x])

    def words_to_ids(self, x: Iterable[str]) -> List[int]:
        return [self._w2i.get(i, self.unk) for i in x]

    def ids_to_words(self, x: Iterable[int]) -> List[str]:
        return [self._i2w[i] for i in x]

    def _tokenize(self, x: str) -> List[str]:
        return self._tokenizer(x.lower())

    def _clean_token(self, x: str) -> str:
        return self._clean_re("", x)

    def tokenize(
        self,
        x: Iterable[str] | str,
        char_seq: bool = True,
        word_seq: bool = True,
        aligned: bool = True,
    ) -> List[int] | Tuple[List[int], List[int]]:
        """
        Args:
            x: string to tokenize
            char_seq: return the sequence of characters
            word_seq: return the sequence of words
            aligned: align the word sequence and the char sequence by repeating the word tokens
        Return:
            char sequence, word sequence or both
        """
        if not char_seq and not word_seq:
            raise ValueError("Either char_seq or word_seq must be True")

        if isinstance(x, str):
            x = self._tokenize(x)
        x = map(self._clean_token, x)

        ids = [([self.bos], self.bos)]  # List of tuples with char ids and word id
        for token in x:
            tmp = (self.words_to_ids(token), self.chars_to_ids(token))
            ids.append(tmp)
        ids.append(([self.eos], self.eos))

        if char_seq and not word_seq:
            # Flatten char ids and add spacess
            chars = []
            for i in ids:
                chars.extend(i[0])
                chars.append(0)  # Use space as pad
            return chars
        elif not char_seq and word_seq:
            # Seq of word ids
            return [i[1] for i in ids]
        elif not aligned:
            chars = []
            words = []
            for i in ids:
                chars.extend(i[0])
                chars.append(self.pad)  # Use space as pad
                words.append(i[1])
            return chars, words
        else:
            # Align
            chars = []
            words = []
            for i in ids:
                chars.extend(i[0])
                chars.append(self.pad)
                words.extend([i[1] for _ in i[0]])
                words.append(self.pad)
            return chars, words

    def batch_tokenize(
        self,
        x: List[str],
        char_seq: bool = True,
        word_seq: bool = True,
        aligned: bool = True,
    ) -> List[int] | Tuple[List[int], List[int]]:
        chars = []
        words = []
        for i in x:
            tmp = self.tokenize(i, char_seq, word_seq, aligned)
            if char_seq and word_seq:
                chars.append(tmp[0])
                words.append(tmp[1])
            elif not word_seq:
                chars.append(tmp)
            else:
                words.append(tmp)

        chars = self.pad_sequences(chars)
        words = self.pad_sequences(words)

        if char_seq and word_seq:
            return chars, words
        if char_seq:
            return chars
        if word_seq:
            return words

    def decode(
        self, x: List[List[int]] | List[int], word: bool = True
    ) -> str | List[str]:
        # TODO
        if word:
            pass
        pass

    def detokenize(
        self, x: List[List[int]] | List[int], word: bool = True
    ) -> str | List[str]:
        return self.decode(x, word)

    def pad_sequences(self, seq: List[List[int]]):
        lengths = [len(i) for i in seq]
        max_len = max(lengths)
        pad = self.pad
        return [s + [pad] * (max_len - l) for s, l in zip(seq, lengths)]

    def vocab_size(self, words: bool = True) -> int:
        if words:
            return len(self._w2i)
        return len(self._c2i)


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, alphabet: List[str]) -> None:
        """
        SentencePiece tokenizer used for the neural networks.

        Args:
            alphabet: characters used by the tokenizer
        """
        super().__init__()
        self._sp_words = spm.SentencePieceProcessor()
        self._sp_chars = spm.SentencePieceProcessor()
        self._alphabet = alphabet
        self._clean_re = re.compile(rf"[^{''.join(alphabet)}]")
        self._space_norm = re.compile(r" +")

    def train(
        self,
        txt: str | List[str],
        name: str,
        model_type: str = "unigram",
        vocab_size: int = 2000,
    ) -> None:
        """
        Train the model and save it.
        This generate the files `{name}_word.model`, `{name}_word.vocab`, `{name}_char.model`, `{name}_char.vocab`.

        Args:
            txt: path to the training data or list of training sentences
            name: name of the mode
            model_type: bpe, unigram, word

        Returns:
            None
        """
        # Load lines
        if isinstance(txt, str):  # path
            with open(txt) as f:
                txt = f.readlines()

        txt = [i.strip().lower() for i in txt]
        txt = [self._clean_re.sub(" ", i) for i in txt]

        # pad=0 is easier to integrate in the model
        # Decode <unk> as <unk>
        # special_tokens = "--pad_id=0 --unk_id=3 --bos_id=1 --eos_id=2 --pad_piece=<pad> --unk_piece=<unk> --bos_piece=<bos> --eos_piece=<eos> --unk_surface=<unk>"

        # Word sequence
        # HACK remove demaged tokens -> all tokens with - are tokenized as <unk>
        # This migth not work with BPE
        word_lines = [re.sub(r"(\W|^)(\w*-+\w*)+(\W|$)", " ", i) for i in txt]
        word_lines = [self._space_norm.sub(" ", i) for i in txt]
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(word_lines),
            vocab_size=vocab_size,
            model_prefix=f"{name}_word",
            model_type=model_type,
            pad_id=0,
            pad_piece="<pad>",
            bos_id=1,
            bos_piece="<s>",
            eos_id=2,
            eos_piece="</s>",
            unk_id=3,
            unk_piece="<unk>",
            unk_surface="<unk>",
            user_defined_symbols="▁",  # NOTE: not _ but ▁
        )
        # Char sequence
        # HACK - to <unk> <- - not in the training data
        char_lines = [i.replace("-", " ") for i in txt]
        char_lines = [self._space_norm.sub(" ", i) for i in txt]
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(char_lines),
            vocab_size=len(self._alphabet) + 5,  # Special tokens
            model_prefix=f"{name}_char",
            model_type="char",
            pad_id=0,
            pad_piece="<pad>",
            bos_id=1,
            bos_piece="<s>",
            eos_id=2,
            eos_piece="</s>",
            unk_id=3,
            unk_piece="<unk>",
            unk_surface="-",
            user_defined_symbols="▁",  # NOTE: not _ but ▁
        )

    def load(self, model: str) -> Self:
        """
        Load the SentencePiece models given the name.
        This requires the `{model}_word.model` and `{model}_char.model` files.

        Args:
            model: path / name of the model

        Return:
            Itself (not necessary)
        """
        # TODO: return None instead
        self._sp_words.load(f"{model}_word.model")
        self._sp_chars.load(f"{model}_char.model")
        return self

    def tokenize(
        self, x: str | List[str], align: bool = True, align_mode: str = "same"
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Tokenize a sentence of a list of sentences

        Args:
            x: text to tokenize
            align: if to align the character and word sequences
            align_mode: how to align the sequences ("same" or "space")

        Returns:
            List of character sequences and list of word sequences
        """
        if isinstance(x, str):
            x = [x]

        # No alignment -> tokenize independently
        if not align:
            return self._sp_chars.encode_as_ids(
                x, add_bos=True, add_eos=True
            ), self._sp_words.encode_as_ids(x, add_bos=True, add_eos=True)

        # Alignment -> repeat word tokens

        # [[id id ...], [id id ...]]
        char_seq = self._sp_chars.encode_as_ids(x, add_bos=True, add_eos=True)

        # [[tok tok ...], [tok tok ...]], _ is separated
        tmp_word_seq = self._sp_words.encode_as_pieces(x, add_bos=True, add_eos=True)
        word_seq = [[] for _ in range(len(x))]

        if align_mode == "same":
            get_id = lambda x: self._sp_words.piece_to_id(x)
        else:
            space_id = self._sp_words.piece_to_id("▁")  # NOTE: not _
            get_id = lambda _: space_id

        for i, s in enumerate(tmp_word_seq):
            for tok in s:
                id_ = get_id(tok)
                if self._sp_words.is_control(id_):
                    # Control/-> length 1
                    word_seq[i].append(id_)
                else:
                    # Normal token / unk -> repeat for each char in the token
                    word_seq[i].extend([id_] * len(tok))
        return char_seq, word_seq

    def decode(
        self, x: List[List[int]] | List[int], word: bool = True
    ) -> str | List[str]:
        """
        Detokenize one or more token sequences (word or characters).

        Args:
            x: sequences
            word: if it is a word sequence

        Returns:
            List of detokenized text (or str if it only one).

        Note: it is not possible to detokenize word sequences aligned as "same"
        """
        return self.detokenize(x, word)

    def detokenize(
        self, x: List[List[int]] | List[int], word: bool = True
    ) -> str | List[str]:
        """
        Detokenize one or more token sequences (word or characters).

        Args:
            x: sequences
            word: if it is a word sequence

        Returns:
            List of detokenized text (or str if it only one).

        Note: it is not possible to detokenize word sequences aligned as "same"
        """
        if not isinstance(x, list):
            # jax.Array or np.array
            x = x.tolist()
        if word:
            # blank = self._sp_words.piece_to_id("▁")
            # tmp = []
            # for i in x:
            #     tmp.append(list(chain.from_iterable(zip(i, [blank] * len(i)))))
            return self._sp_words.decode(x)
        return self._sp_chars.decode(x)

    def vocab_size(self, words: bool = True) -> int:
        """
        Vocabulary size of the SentecePiece model

        Args:
            words: if it is the word model

        Return:
            number of vocables
        """
        if words:
            return self._sp_words.vocab_size()
        return self._sp_chars.vocab_size()
