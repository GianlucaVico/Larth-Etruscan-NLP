"""Class for the random translation model"""
from base_models import BaseModel
import numpy as np
from typing import Callable, Iterable, List, Dict, Any, Self
from nltk import word_tokenize
from itertools import chain
from collections import Counter
import json
import sys

sys.path.append("../")
from Data import TokenizerType


class RandomModel(BaseModel):
    """
    Random translation model. It samples English words from the training translations.

    Args:
        cutoff: ignore tokens whose frequency is less than the cutoff value
        length_dist: distribution of the length of the translation in tokens.
            - normal: assume normal distribution
            - uniform: assume uniform distribution
            - actual: used the same obtain from the training set
        seed: random number generator seed
        name: name of the model
    """

    def __init__(
        self, cutoff: int = 0, length_dist: str = "normal", seed: int = 0, name="random"
    ):
        super().__init__()
        if length_dist not in ["normal", "uniform", "actual"]:
            raise ValueError(
                f"length_dist must be either 'normal' or 'uniform' or 'actual', but {length_dist} found"
            )

        self._name: str = name

        self._cutoff: int = cutoff

        self._vocab: np.array = None
        self._probs: np.array = None
        self._lengths: np.array = None

        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._random_len: Callable[[None], int]
        self._length_dist: str = length_dist

        self._dist_map = {
            "normal": self._est_len_norm,
            "uniform": self._est_len_uniform,
            "actual": self._est_len_actual,
        }

    @property
    def cutoff(self) -> int:
        """
        Returns:
            cutoff
        """
        return self._cutoff

    def train(
        self,
        english: Iterable[str],
        english_tokenizer: TokenizerType = word_tokenize,
    ) -> None:
        """
        Train the model.

        Args:
            english: list of translations
            english_tokenizer: tokenizer use to split the sentences
        """
        english = [english_tokenizer(i) for i in english]
        self._lengths = np.array([len(i) for i in english])

        english = chain.from_iterable(english)

        self._random_len = self._dist_map[self._length_dist](self._lengths)

        c = Counter(english)
        c = [(i, j) for i, j in c.items() if j >= self._cutoff]

        vocab, weights = zip(*c)

        self._vocab = np.array(vocab)
        self._probs = np.array(weights)
        self._probs = self._probs / self._probs.sum()

    def predict(self, x: str | Iterable[str]) -> str | List[str]:
        """
        Generate a "translation"

        Args:
            x: input string or list of strings

        Returns:
            String for each string in input
        """
        if isinstance(x, str):
            return self._predict_single()
        else:
            return [self._predict_single() for _ in x]

    def _predict_single(self) -> str:
        """
        Returns:
            A random translation
        """
        l = self._random_len()
        sent = self._rng.choice(self._vocab, l, True, self._probs)
        return " ".join(sent)

    def _est_len_norm(self, lengths: np.array) -> Callable[[None], int]:
        """
        Estimate normal distribution

        Args:
            lengths: lengths the training translations

        Returns:
            Function that samples the estimated distribution
        """
        mean = np.mean(lengths)
        std = np.std(lengths)

        def f():
            return int(np.round(np.abs(self._rng.normal(mean, std))))

        return f

    def _est_len_uniform(self, lengths: np.array) -> Callable[[None], int]:
        """
        Estimate uniform distribution

        Args:
            lengths: lengths the training translations

        Returns:
            Function that samples the estimated distribution
        """
        min_ = np.min(lengths)
        max_ = np.max(lengths)

        def f():
            return int(np.round(self._rng.uniform(min_, max_)))

        return f

    def _est_len_actual(self, lengths: np.array) -> Callable[[None], int]:
        """
        Sample the translation lengths

        Args:
            lengths: lengths the training translations

        Returns:
            Function that samples the estimated distribution
        """

        def f():
            return self._rng.choice(lengths)

        return f

    def save(self, path: str | None = None) -> Dict[str, Any]:
        """
        Save the model as json file

        Args:
            path: file where to save the model

        Returns:
            Dictionary representation of the model
        """
        v = None
        p = None
        l = None
        if self._vocab is not None:
            v = self._vocab.tolist()
        if self._probs is not None:
            p = self._probs.tolist()
        if self._lengths is not None:
            l = self._lengths.tolist()

        d = {
            "name": self._name,
            "cutoff": self._cutoff,
            "vocab": v,
            "probs": p,
            "lengths": l,
            "seed": self._seed,
            "dist": self._length_dist,
        }
        if path is not None:
            with open(path, "w") as f:
                json.dump(d, f)
        return d

    @staticmethod
    def load(path: Dict[str, Any] | str) -> Self:
        """
        Load the model

        Args:
            path: path to a json file or a dictionary representing the model

        Returns:
            the loaded model
        """
        if isinstance(path, str):
            with open(path, "r") as f:
                d = json.load(f)
        else:
            d = path

        model = RandomModel(d["cutoff"], d["dist"], d["seed"], d["name"])

        if d["vocab"] is not None:
            model._vocab = np.array(d["vocab"])
        if d["probs"] is not None:
            model._probs = np.array(d["probs"])
        if d["lengths"] is not None:
            model._lengths = np.array(d["lengths"])
            model._random_len = model._dist_map[model._length_dist](model._lengths)

        return model
