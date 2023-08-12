"""
Base classes for the translation models
"""
from abc import ABC, abstractmethod
from typing import (
    Any,
    Self,
    Dict,
)


class BaseModel(ABC):
    """
    Base class for translation models
    """

    _name: str = "base_model"

    def __call__(self, x: str, *args) -> str:
        return self.predict(x, *args)

    @abstractmethod
    def predict(self, x: str, *args) -> str:
        """
        Translate the input sentence

        Args:
            x: Etruscan sentence

        Returns:
            English translation
        """
        pass

    @abstractmethod
    def save(self, path: str | None = None) -> Dict[str, Any]:
        """
        Save the model in json format

        Args:
            path: path where to store the model

        Returns:
            Dictionary representing the model
        """
        pass

    @staticmethod
    @abstractmethod
    def load(path: Dict[str, Any] | str) -> Self:
        """
        Load the model from a file or dictionary

        Args:
            path: either a path to a json file or a dictionary

        Returns:
            A new model
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns:
            Name identifying the model
        """
        return self._name
