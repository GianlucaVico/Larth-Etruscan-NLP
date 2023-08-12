"""
Translate Etruscan with a dictionary
"""
from typing import Any, Dict, Self, Callable
import base_models
import json

import sys

sys.path.append("../")
sys.path.append("../..")
import Data
import Data.utils as utils


class DictionaryTranslation(base_models.BaseModel):
    _name: str = "dictionary_model"
    """
    Dictionary-base translation model. 
    It tokenizes the Etruscan sentence and then translates one toke at the time.

    Args:
        dictionary: dictionary to translate the text, {"etruscan": "english"}
        tokenizer: tokenizer to split the Etruscan text
        unk: translate unknown tokens to this
        etruscan_fn: function to process the vocabulary file
        english_fn: function to process the vocabulary file
        tokenize_dictionary: function to clean and tokenize the Etruscan words in the dictionary
    """

    def __init__(
        self,
        dictionary: str | Dict[str, str],
        tokenizer: Data.TokenizerType = Data.BlankspaceTokenizer(),
        unk: str | None = None,
        etruscan_fn: Callable[[str], str] = None,
        english_fn: Callable[[str], str] = None,
        tokenize_dictionary: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer

        if unk is not None:
            self.unk = unk
        else:
            self.unk = ""

        # If it is already a dictionary, use it as it is
        # Otherwise read the POS csv file
        if isinstance(dictionary, str):
            dict_df = Data.load_pos(dictionary)[["Etruscan", "Translations"]].dropna()
            # Use only the firs translation
            keys = dict_df["Etruscan"].apply(lambda x: utils.replace(x, utils.to_latin))
            if etruscan_fn is not None:
                keys = keys.apply(etruscan_fn)
            values = dict_df["Translations"].apply(
                lambda x: "" if len(x) == 0 else x[0][1]
            )
            if english_fn is not None:
                values = values.apply(english_fn)

            keys, values = keys.to_list(), values.to_list()
            if tokenize_dictionary is not None:
                keys = [tokenize_dictionary(i) for i in keys]
            d = dict(zip(keys, values))
        else:
            d = dictionary.copy()

        # Remove empty entries
        d = {k: v for k, v in d.items() if len(k) != 0 and len(v) != 0}
        self.dict = d

    def predict(self, x: str) -> str:
        """
        Translate a single Etruscan string into English

        Args:
            x: Etruscan text

        Returns:
            English translation
        """
        x = self.tokenizer(x)
        translation = [self.dict.get(i, self.unk) for i in x]
        return " ".join(translation)

    def save(self, path: str | None = None) -> Dict[str, Any]:
        """
        Save the model as a dict and in a json file

        Args:
            path: if given, where to save the model

        Returns:
            Dictionary representin the model

        Notes:
            It can't save the tokenizers
        """
        d = {"unk": self.unk, "dict": self.dict, "name": self._name}
        if not isinstance(self.tokenizer, Data.BaseTokenizer):
            print("Can't save the tokenizer: not supported")

        if path is not None:
            with open(path, "wt") as f:
                json.dump(d, f)

        return d

    @staticmethod
    def load(path: Dict[str, Any] | str) -> Self:
        """
        Load the model from a file or a dictionary

        Args:
            path: path to a json file or a dictionary

        Returns:
            Restored model

        Notes:
            The tokenizer is the default one
        """
        if isinstance(path, str):
            with open(path) as f:
                d = json.load(f)
        else:
            d = path

        model = DictionaryTranslation(d["dict"], Data.BlankspaceTokenizer(), d["unk"])
        model._name = d["name"]
        return model
