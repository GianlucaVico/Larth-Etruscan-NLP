"""
N-gram language models
"""
import sys

sys.path.append("..")

from Translation.base_models import BaseModel
from Data import TokenizerType
from nltk.lm import models
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import lm
from nltk.util import pad_sequence, ngrams
import pickle
import json

from typing import Dict, Any, Self, List, Callable, Tuple
import numpy as np


def LMWrapper(model: models.LanguageModel, **kargs) -> Callable:
    """Generate a wrapper for NLTK LMs"""

    def f(order: int):
        return model(order=order, **kargs)

    return f


class NgramLmModel(BaseModel):
    """
    N-gram language model for Etruscan
    """

    _name: str = "ngram_lm_model"

    def __init__(
        self,
        n: int,
        tokenizer: TokenizerType,
        base_model: Callable[[int], models.LanguageModel] = lm.MLE,
    ) -> None:
        """
        Args:
            n: order of the model
            tokenizer: tokenizer
            base_model: NLTK language model. Use `LMWrapper` to pass its parameters if needed
        """
        super().__init__()
        self._n = n
        self._tokenizer = tokenizer
        self._lm = base_model(n)

    def predict(
        self,
        x: str | List[str] | None,
        seed: int = 0,
        n: int = 1,
        stop_eos: bool = False,
        join: str = " ",
        *args
    ) -> str:
        """
        Predict the next tokens

        Args:
            x: input text as string or list of tokens
            seed: rng seed
            n: number of tokens generated
            stop_eos: whether to stop when EOS is generated
            join: space delimiter

        Returns:
            Predicted string (including the input)
        """
        # Tokenize each example if not already tokenized
        if isinstance(x, str):
            x = self._tokenizer(x)

        # Not needed
        # x = list(pad_sequence(x,
        # pad_left=True, left_pad_symbol="<s>",
        # pad_right=False, n=self.n))
        if n == 1:
            generated = x + [
                self._lm.generate(num_words=n, text_seed=x, random_seed=seed)
            ]
        else:
            generated = x + self._lm.generate(
                num_words=n, text_seed=x, random_seed=seed
            )

        if stop_eos:
            index = None
            try:
                index = generated.index("</s>")
            except ValueError:
                pass
            if index is not None:
                generated = generated[:index]

        # Detokenize
        remove = ["<s>", "</s>", "<unk>", "<UNK>"]
        generated = [i.lower() for i in generated if i.lower() not in remove]
        return self._tokenizer.detokenize(generated)

    def save(self, path: None | str = None) -> Dict[str, Any]:
        """
        Save the model.

        Args:
            path: json file where to save

        Returns:
            Dictionary representation of the model

        Note: this also generate a pickle file for the NLTK lm
        """
        d = {
            "name": self.name,
            "n": self.n,
            "lm": None,
        }

        if path is not None:
            # Save tokenizer and lm with pickle
            lm_path = path.split(".")[0] + "_lm.pickle"
            with open(lm_path, "w") as f:
                pickle.dump((self._lm, self._tokenizer), f)
            d["lm"] = lm_path
            with open(path, "w") as f:
                json.dump(d, f)
        return d

    @staticmethod
    def load(path: Dict[str, Any] | str) -> Self:
        """
        Load the model from a file.

        Args:
            path: path to a json file or dictionary representation of the model

        Returns:
            Loaded model
        """
        with open(path) as f:
            d = json.load(f)
        tok, model = None
        if d["lm"] is not None:
            with open(d["lm"]) as f:
                tok, model = pickle.load(f)
        ngram = NgramLmModel(d["n"], tok, lm.MLE)
        ngram._name = d["name"]
        ngram._lm = model
        return ngram

    @property
    def n(self) -> int:
        """
        Returns:
            Order of the model
        """
        return self._n

    def train(self, sentences: List[List[str]] | List[str]) -> None:
        """
        Train the model.

        Args:
            sentences: list of training sentences (can be already tokenized)
        """
        # Tokenize each example if not already tokenized
        if isinstance(sentences[0], str):
            sentences = [self._tokenizer(i) for i in sentences]

        train, vocab = padded_everygram_pipeline(self.n, sentences)
        self._lm.fit(train, vocab)

    def perplexity(self, sentence: List[str] | str) -> float:
        """
        Compute the perplexity of the model

        Args:
            sentence: test sentence (can be already tokenized)

        Returns:
            Perplexity
        """
        if isinstance(sentence, str):
            sentence = self._tokenizer(sentence)

        sentence = list(
            pad_sequence(
                sentence,
                pad_left=True,
                left_pad_symbol="<s>",
                pad_right=True,
                right_pad_symbol="</s>",
                n=self.n,
            )
        )

        # List of ngrams [(1,2,3), (2,3,4), etc]
        ng = list(ngrams(sentence, self.n))
        return self._lm.perplexity(ng)


def compute_perplexity(
    sentences: List[str], model: NgramLmModel
) -> Tuple[float, float]:
    """
    Compute the average perplexity of the model.

    Args:
        sentences: list of test sentences
        model: model to test

    Returns:
        mean and standard deviation of the perplexity.
    """
    tmp = [model.perplexity(i) for i in sentences]
    return np.mean(tmp), np.std(tmp)
