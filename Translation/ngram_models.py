"""
Module for the the NGram translation model.
"""
from nltk.lm import Vocabulary
from base_models import BaseModel

import nltk
from nltk.probability import (
    ConditionalFreqDist,
    ConditionalProbDist,
    FreqDist,
    LidstoneProbDist,
)

import json
from dataclasses import asdict, dataclass
from itertools import chain
from typing import Any, Dict, FrozenSet, Iterable, List, Self, Tuple

import sys

sys.path.append("../")
from Data import TokenizerType


@dataclass(frozen=True)
class NGramModelSettings:
    """
    Parameters for the NGramModel

    Args:
        n: ngram size
        gamma: smooting factor for the Lidstone distribution. The smoothing works as $\dfrac{count + gamma}{total + voc. size * gamma}$
        eng_vocab: vocabulary with English words
        et_vocab: vocabulary with Etruscan words
        ignore_order: whether to ignore the word order in the ngrams
        english_etruscan_context: if True the context is given by the next Etruscan ngram and the English ngram just translated. Otherwise, only the Etruscan context is used
        unk: unknown token (default: <UNK>)
        left_pad: token to pad the sequence at the beginning (default: <s>)
        right_pad: token to pad the sequence at the end (default: </s>)
        use_bayes: if True, $p(eng | context)$ is estimated as $p(eng) p(context_1 | eng) p(context_2 | eng) ...$. Otherwise, it is estimated directly.
    """

    n: int = 2
    gamma: float = 1
    eng_vocab: Vocabulary | None = None
    et_vocab: Vocabulary | None = None
    ignore_order: bool = False
    english_etruscan_context: bool = False
    unk: str = "<unk>"
    left_pad: str = "<s>"
    right_pad: str = "</s>"
    use_bayes: bool = False

    def __repr__(self) -> str:
        ord = ""
        if self.ignore_order:
            ord = " - no order - "
        bayes = ""
        if self.use_bayes:
            bayes = " - naive bayes - "
        context = " - context:et - "
        if self.english_etruscan_context:
            context = " - context:et-eng - "
        return f"n:{self.n}{ord}{context}{bayes}"

    def __str__(self) -> str:
        return str(dict(self))


# Possible context of the NGramModel
ContextType = (
    FrozenSet[str]
    | Tuple[str]
    | Tuple[Tuple[str], Tuple[str]]
    | Tuple[FrozenSet[str], FrozenSet[str]]
)


class NGramModel(BaseModel):
    _name: str = "ngram_model"

    def __init__(self, settings: NGramModelSettings) -> None:
        """
        NGram model that translate an Etruscan sentence into English by
        estimating the probability distribution p(english token | context)

        Args:
            settings: settings of the model
        """
        super().__init__()
        self._settings: NGramModelSettings = settings

        # P(eng | context)
        self._p_eng_given_context: ConditionalProbDist = None
        # P(eng)
        self._p_eng: LidstoneProbDist = None

    @property
    def gamma(self) -> float:
        """
        Return:
            g: smoothing factor for the prob. distribution
        """
        return self._settings.gamma

    @property
    def n(self) -> int:
        """
        Returns:
            n
        """
        return self._settings.n

    def _convert_ngrams(
        self, sentences: List[List[str]]
    ) -> List[List[FrozenSet[str]]] | List[List[Tuple[str]]]:
        """
        Turn a list of sentences into ngrams of the appropriate type

        Args:
            sentences: list of tokenized sentences

        Returns:
            List of list of ngrams. The ngrams can be tuples of frozen sets.
        """
        # Use sets (frozenset -> hashable)
        if self._settings.ignore_order:
            # For each example convert the ngrams to sets
            out = [[frozenset(i) for i in eng] for eng in sentences]
        else:
            # Use tuple otherwise
            out = [[tuple(i) for i in eng] for eng in sentences]
        return out

    def _context_iterator(self, context: ContextType) -> Iterable:
        """
        Allow to iterate through the context.

        If the context is only Etruscan, it does nothing.
        Otherwise, it gives the pairs of English and Etruscan tokens.

        Note: this method is not meaningful when ignore order is True.
        """
        if self._settings.english_etruscan_context:
            context = zip(*context)
        return context

    def train(
        self,
        etruscan: Iterable[str],
        english: Iterable[str],
        etruscan_tokenizer: TokenizerType,
        english_tokenizer: TokenizerType,
    ) -> None:
        """
        Method to train the model.

        Args:
            etruscan: list of Etruscan sentences
            english: list of English sentences
            etruscan_tokenizer: tokenizer to precess Etruscan sentences
            english_tokenizer: tokenizer to precess English sentences
        """
        # Tokenize
        etruscan = [etruscan_tokenizer(i) for i in etruscan]
        english = [english_tokenizer(i) for i in english]

        # Clean text
        if self._settings.eng_vocab is not None:
            english = [self._settings.eng_vocab.lookup(i) for i in english]
        if self._settings.et_vocab is not None:
            etruscan = [self._settings.et_vocab.lookup(i) for i in etruscan]
        # Pad
        for i in range(len(etruscan)):
            et_len = len(etruscan[i])
            eng_len = len(english[i])

            # Pad for the n-grams and to get the same length
            etruscan[i] = list(
                nltk.pad_sequence(
                    etruscan[i],
                    self.n + (max(0, eng_len - et_len)),
                    pad_right=True,
                    pad_left=True,
                    left_pad_symbol=self._settings.left_pad,
                    right_pad_symbol=self._settings.right_pad,
                )
            )
            english[i] = list(
                nltk.pad_sequence(
                    english[i],
                    self.n + (max(0, et_len - eng_len)),
                    pad_right=True,
                    pad_left=True,
                    left_pad_symbol=self._settings.left_pad,
                    right_pad_symbol=self._settings.right_pad,
                )
            )

        # Used to train with English context
        # Use the previous English ngram
        shift_english = [[self._settings.left_pad] * self.n + i for i in english]

        # To n-grams
        n_etruscan = [nltk.ngrams(i, self.n) for i in etruscan]
        n_english = [nltk.ngrams(i, self.n) for i in english]
        n_shift_english = [nltk.ngrams(i, self.n) for i in shift_english]

        # To frozenset / tuple
        n_etruscan = self._convert_ngrams(n_etruscan)
        n_english = self._convert_ngrams(n_english)
        n_shift_english = self._convert_ngrams(n_shift_english)

        # Flatten everything
        english = list(chain.from_iterable(english))
        shift_english = list(chain.from_iterable(shift_english))
        etruscan = list(chain.from_iterable(etruscan))
        n_english = list(chain.from_iterable(n_english))
        n_shift_english = list(chain.from_iterable(n_shift_english))
        n_etruscan = list(chain.from_iterable(n_etruscan))

        # For each element in etruscan, count the english n-grams
        if self._settings.english_etruscan_context:
            context = list(zip(n_etruscan, n_shift_english))
        else:
            context = n_etruscan

        if self._settings.use_bayes:
            context_token_pairs = []
            for c, e in zip(context, english):  # Pairs context, eng token
                for i in self._context_iterator(c):  # Tokens in the context
                    context_token_pairs.append((e, i))

            freq = ConditionalFreqDist(context_token_pairs)
        else:
            freq = ConditionalFreqDist(zip(context, english))  # Context, token

        self._p_eng_given_context = ConditionalProbDist(
            freq, LidstoneProbDist, **{"gamma": self.gamma}
        )

        freq = FreqDist(english)
        self._p_eng = LidstoneProbDist(freq, self.gamma)

    def predict(
        self,
        etruscan: str,
        etruscan_tokenizer: TokenizerType,
        max_len: int | None = None,
        beam: int | None = 8,
    ) -> Tuple[str, float]:
        """
        Translate an Etruscan sentence into English.

        Args:
            etruscan: Etruscan sentence
            etruscan_tokenizer: tokenizer for the Etruscan sentence
            max_len: force the translation to be at least of this length (not used when `english_etruscan_context` is True)
            beam: number of beams (not used when `english_etruscan_context` is False)

        Returns:
            Translation and its log-probability
        """
        if not self._settings.english_etruscan_context:
            return self._greedy_decoding(etruscan, etruscan_tokenizer, max_len)
        else:
            return self._beam_search_decoding(etruscan, etruscan_tokenizer, beam)

    def _greedy_decoding(
        self,
        etruscan: str,
        etruscan_tokenizer: TokenizerType,
        max_len: int | None = None,
    ) -> Tuple[str, float]:
        """
        Method to translate a sentence when english_etruscan_context is true.

        Args:
            etruscan: Etruscan sentence
            etruscan_tokenizer: tokenizer for the Etruscan sentence
            max_len: force the translation to be at least of this length

        Returns:
            Translation and its log-probability
        """
        right_empty_context = [[self._settings.right_pad] * self.n]
        right_empty_context = self._convert_ngrams([right_empty_context])[0][0]

        etruscan = etruscan_tokenizer(etruscan)
        etruscan = list(
            nltk.pad_sequence(
                etruscan,
                self.n,
                pad_left=True,
                pad_right=True,
                left_pad_symbol=self._settings.left_pad,
                right_pad_symbol=self._settings.right_pad,
            )
        )
        n_etruscan = list(nltk.ngrams(etruscan, self.n))
        n_etruscan = self._convert_ngrams([n_etruscan])[0]

        seq = []
        logprob = 0
        for ngram in n_etruscan:
            token, prob = self._predict_next(ngram)
            logprob += prob[0]
            seq.append(token[0])
            # print(token, "|", *ngram)
        if (
            seq[-1] != self._settings.right_pad
            and max_len is not None
            and len(seq) < max_len
        ):
            for _ in range(max_len - len(seq)):
                token, prob = self._predict_next(right_empty_context)
                logprob += prob[0]
                seq.append(token[0])

                if token[0] == self._settings.right_pad:
                    break
        return " ".join(seq), logprob

    def _beam_search_decoding(
        self,
        etruscan: str,
        etruscan_tokenizer: TokenizerType,
        beam: int = 8,
    ) -> Tuple[str, float]:
        """
        Beam search to translate a sentence.

        Args:
            etruscan: Etruscan sentence
            etruscan_tokenizer: tokenizer for the Etruscan sentence
            beam: number of beams

        Returns:
            Best translation and its log-probability
        """

        # Fn to sort the candidates
        sort_key = lambda x: x[1]

        # Prepare the Etruscan sequence
        right_empty_context = [[self._settings.right_pad] * self.n]
        right_empty_context = self._convert_ngrams([right_empty_context])[0][0]
        left_empty_context = [[self._settings.left_pad] * self.n]
        left_empty_context = self._convert_ngrams([left_empty_context])[0][0]

        etruscan = etruscan_tokenizer(etruscan)
        etruscan = list(
            nltk.pad_sequence(
                etruscan,
                self.n,
                pad_left=True,
                pad_right=True,
                left_pad_symbol=self._settings.left_pad,
                right_pad_symbol=self._settings.right_pad,
            )
        )
        n_etruscan = list(nltk.ngrams(etruscan, self.n))
        n_etruscan = self._convert_ngrams([n_etruscan])[0]

        # List of candidate solution
        candidates = [
            ([token], prob)
            for token, prob in zip(
                *self._predict_next((n_etruscan[0], left_empty_context), k=beam)
            )
        ]
        tmp = [None] * (beam**2)  # Store the successors

        # Translate each ngram
        for ngram in n_etruscan[1:]:
            # Continue the translation of each candidate
            for i, (seq, prob) in enumerate(candidates):
                if len(seq) < self.n:
                    seq = [self._settings.left_pad for _ in range(self.n)]
                context = (ngram, tuple(seq[-self.n :]))
                successors = zip(*self._predict_next(context, k=beam))

                # Generate the new sequences
                for j, (token, p) in enumerate(successors):
                    new_prob = prob + p
                    tmp[i * beam + j] = (seq + [token], new_prob)

            # Prune
            tmp = sorted(tmp, key=sort_key, reverse=True)
            candidates = tmp[:beam]

        # Final translation
        translation = " ".join(candidates[0][0])
        return translation, candidates[0][1]

    def _predict_next(
        self, context: ContextType, k: int = 1
    ) -> Tuple[List[str], List[float]]:
        """
        Find the top-k probable tokens given the context.

        Args:
            context: context of the appropriate type
            k: number of samples to return

        Returns:
            List of tokes and list of token's probabilities

        Note:
            This method can return the unknown token.
        """
        if self._settings.use_bayes:
            d = (
                []
            )  # Log-probabilities P(eng) P(context 1 | eng) * P(context 2 | eng) * ...
            for c in self._p_eng_given_context.conditions():
                if c != self._settings.left_pad:
                    # c: english token
                    dist = self._p_eng_given_context[c]
                    # Iterate through the context tokens

                    logprob = 0
                    for i in self._context_iterator(context):
                        logprob += dist.logprob(i)

                    d.append((c, logprob + self._p_eng.logprob(c)))

            # Take the most probable
            d = sorted(d, key=lambda x: x[1], reverse=True)
            # Return top k
            return tuple(zip(*d[:k]))
        else:
            # Return the n most probable tokens givens the context
            cond_dist = self._p_eng_given_context.get(context)

            if cond_dist is None:
                # Unknown context
                # Return most probable English tokens
                cond_dist = self._p_eng

            # Find the most probable tokens
            # Ignore the count, use only the token
            top_k = cond_dist.freqdist().most_common(k)
            top_k = top_k + [(self._settings.unk, 0)] * max(
                k - len(top_k), 0
            )  # Fill top_k if necessary

            tokens = [i[0] for i in top_k]
            probs = [cond_dist.logprob(i[0]) for i in top_k]
            return tokens, probs

    @staticmethod
    def load(path: Dict[str, Any] | str) -> Self:
        """
        Load a model from a json file or a dictionary

        Args:
            path: path to the json file or a dictionary containing the model

        Returns:
            A new model
        """
        if isinstance(path, str):
            with open(path, "r") as f:
                d = json.load(f)
        else:
            d = path

        settings = NGramModelSettings(**d["settings"])
        model = NGramModel(settings)
        model._name = d["name"]

        freq = FreqDist(d["p_eng"])
        model._p_eng = LidstoneProbDist(freq, settings.gamma)

        freq = ConditionalFreqDist()
        tmp = d["p_eng_given_context"]
        tmp = {i: FreqDist(j) for i, j in tmp}  # Context: English distribution
        freq.update(tmp)  # We can't create the object directly
        model._p_eng_given_context = ConditionalProbDist(
            freq, LidstoneProbDist, **{"gamma": settings.gamma}
        )

        return model

    def save(self, path: str | None = None) -> Dict[str, Any]:
        """
        Save the model to a json file and return it as dicitonary
        Args:
            path: path to the file where to store the model
        Returns:
            Dictionary representing the model
        """

        def prob_dist_to_dict(dist: LidstoneProbDist) -> Dict[str, int]:
            return dict(dist.freqdist())

        def cond_dist_to_dict(dist: ConditionalProbDist) -> Dict[str, Dict[str, int]]:
            cond = dist.conditions()
            return {i: prob_dist_to_dict(dist[cond]) for i in cond}

        d = {
            "name": self._name,
            "settings": asdict(self._settings),
            "p_eng": prob_dist_to_dict(self._p_eng),
            "p_eng_given_context": cond_dist_to_dict(self._p_eng_given_context),
        }

        if path is not None:
            with open(path, "w") as f:
                json.dump(d, f)
        return d


def clean_pad(x: str, settings: NGramModelSettings, keep_unk: bool = False) -> str:
    """
    Remove the pad tokens from a string

    Args:
        x: string
        settings: settings of the model that generated the string
        keep_unk: whether to keep the unkown token

    Returns:
        Clean string
    """
    x = x.replace(settings.left_pad, "")
    x = x.replace(settings.right_pad, "")
    if not keep_unk:
        x = x.replace(settings.unk, "")
    return x
