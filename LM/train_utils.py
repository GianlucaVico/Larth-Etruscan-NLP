"""
Common functions for the training
"""
import collections
from typing import List, Dict, Iterator, Tuple
import pyarrow as pa
import jax
import jax.numpy as jnp
import flax.linen as nn
import logging
import numpy as np
from dataclasses import dataclass

from Translation.Larth.train_utils import (
    rsqrt_schedule,
    create_learning_rate_schedule,
    parse_config,
    save_config,
    compute_weighted_cross_entropy,
    compute_weighted_accuracy,
    compute_metrics,
    pad_examples,
    tohost,
    LANGUAGES,
    DATASET_TYPES,
    ALIGNMENTS,
)


@dataclass
class TrainConfig:
    """
    All the paramenters for training the model and loading the dataset
    """

    batch_size: int = 16
    lr: float = 5e-3
    warmup_steps: int = 1
    weight_decay: float = 5e-4
    workdir: str = "log/"
    label_smoothing: float = 0
    restore_checkpoints: bool = False
    beam_size: int = 1
    length_penalty: float = 0.6
    epochs: int = 1
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 1
    restore_from: str | None = None
    cached: bool = False
    dataset_type: str = "csv"  # csv or tatoeba
    subset: str | None = (
        None  # etp, ciep, both for Etruscan; dev for tatoeba; None for unset
    )
    data_path: str = "../../Data/Etruscan.csv"
    etruscan_vocab: str | None = "../../Data/EPT_POS.csv"
    tokenizer: str = "../../Data/all_small"
    alignment: str = "same"  # Not used
    train_size: float = 0.9
    seed: int = 0
    debug: bool = False
    lang: str = "etruscan"
    min_len: int = 1  # Min length of the training samples
    prompt_len: int = 4  # Number of tokens used for the predictions (including BOS)
    train: bool = True
    eval: bool = True

    use_topk: bool = True
    topk: int = 20
    temperature: float = 1.0

    name_augmentation_max_replacements: int = 0

    unk_augmentation_prob: float = 0.0
    unk_augmentation_len: float = 1.0
    unk_augmentation_iterations: int = 0

    def __post_init__(self):
        # Validate fiels
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be > 0, but {self.batch_size} found")
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be > 0, but {self.lr} found")
        if self.warmup_steps < 1:
            raise ValueError(
                f"Warmup steps must be >= 1, but {self.warmup_steps} found"
            )
        if self.label_smoothing < 0:
            raise ValueError(
                f"Label smoothing must be >= 0, but {self.label_smoothing} found"
            )
        if self.beam_size < 1:
            raise ValueError(f"Beam size must be >= 1, but {self.beam_size} found")
        if self.name_augmentation_max_replacements < 0:
            raise ValueError(
                f"Name augmentation must be >= 0, but {self.name_augmentation_max_replacements} found. Set it to 0 to deactivate"
            )
        if not 0 <= self.unk_augmentation_prob <= 1:
            raise ValueError(
                f"Unk augmentaion probability must be between 0 and 1, but {self.unk_augmentation_prob} found. Set it to 0 to deactivate"
            )
        if self.unk_augmentation_len < 1.0:
            raise ValueError(
                f"Unk augmentaion length must be at least 1 , but {self.unk_augmentation_len} found"
            )
        if self.unk_augmentation_iterations < 0:
            raise ValueError(
                f"Unk augmentaion iteration must be greater than 0, but {self.unk_augmentation_iterations} found. Set it to 0 to deactivate"
            )

        # Check languages
        if self.lang not in LANGUAGES:
            raise ValueError(
                f"Source language must be any of {LANGUAGES}, but {self.lang} found"
            )

        if self.dataset_type not in DATASET_TYPES:
            raise ValueError(
                f"Dataset type must be any of {DATASET_TYPES}, but {self.dataset_type} found"
            )

        if self.alignment not in ALIGNMENTS:
            raise ValueError(
                f"Dataset type must be any of {ALIGNMENTS}, but {self.alignment} found"
            )


###### Data #####
class DataLoader:
    """
    Create the batches and iterate through the datasets.

    Attributes:
        batch_size: batch size
        max_len: maximum length of the sequence (at least 28)
        array_only: returns batches with only jax Arrays
        cached: if true the batches are generated when the dalaloader
            is created, otherwise they are made on the fly
        ds: list of batches (PyArrow batches of dictionaries)
        iterator: batch iterator

    Note: DataLoader can be used as iterator after calling `iter(...)`
    """

    def __init__(
        self,
        ds: pa.Table,
        batch_size: int,
        cached: bool = False,
        max_len: int = 512,
        array_only: bool = True,
    ) -> None:
        """
        Args:
            ds: PyArrow dataset
            batch_size: size of the batches
            cached: whether to immediatly create and store the batched
            max_len: truncate sequences that are longer (minimum 28)
            array_only: returns batches with only arrays and not the original strings
        """
        self.ds: List[pa.RecordBatch] = ds.to_batches(batch_size)
        self.batch_size: int = batch_size
        self._cached: bool = cached
        self._len: int = len(self.ds)

        self.max_len: int = max_len
        self.array_only: bool = array_only

        # Init cache
        if self.cached:
            self.ds = [self.make_batch(i.to_pydict()) for i in self.ds]

    def __next__(self) -> Dict[str, jax.Array]:
        """
        Returns:
            Next batch as dictionary
        """
        d = next(self.iterator)
        if self.cached:
            return d
        return self.make_batch(d.to_pydict())

    def __iter__(self) -> Iterator:
        """
        Return:
            Initialize the dataset iterator and returns itself
        """
        logging.info("DataLoader: creating iterator")
        self.iterator: Iterator = iter(self.ds)
        return self

    def make_batch(self, d: Dict[str, List[List[int]]]) -> Dict[str, jax.Array | str]:
        """
        Create a batch with jax Arrays.

        Args:
            d: batch as a dictionary

        Returns:
            Dict: batch as jax Arrays

            If `array_only` is true, the keys are "source_chars", "target_chars", "target_words",
            with jax Arrays as values.

            If `array_only` is false, there are the additional keys "source" and "target" with a list
            of strings.
        """
        source_words = self.pad(d["source_words"])
        source_words = jnp.array(source_words)[:, : self.max_len]

        if self.array_only:
            return {
                # "source_chars": source_chars,
                "inputs": source_words,
            }
        else:
            return {
                "source": [
                    i[: self.max_len - 2] for i in d["source"]
                ],  # bos and eos tokens
                # "source_chars": source_chars,
                "inputs": source_words,
            }

    def pad(self, l: List[List[int]]) -> jax.Array:
        """
        Pad sequences and cast then to a jax Array.
        Pad with 0

        Args:
            l: list of sequences

        Returns:
            jax Array
        """
        l_lens, _ = self._get_lens(l)
        # max_l = max(self.MIN_LEN, max_l)
        # new_l = jnp.array([np.pad(i[:self.MAX_LEN], (0, max(0, max_l - j))) for i, j in zip(l, l_lens)])
        new_l = self._pad(l, l_lens, self.max_len)
        return new_l

    def _get_lens(self, l: List[List[int]]) -> Tuple[List[int], int]:
        """
        Get the length of the sequences.

        Args:
            l: list of sequences

        Returns:
            List of lengths and maximum length
        """
        l_lens = [len(i) for i in l]
        max_l = max(l_lens)
        return l_lens, max_l

    def _pad(self, l: List[List[int]], lens: List[int], pad_l: int) -> jax.Array:
        """
        Pad a list of sequences.

        Args:
            l: list of sequences
            lens: lengths of the sequences
            pad_l: final length of the sequence

        Returns:
            jax array
        """
        return jnp.array(
            [np.pad(i[: self.max_len], (0, max(0, pad_l - j))) for i, j in zip(l, lens)]
        )

    def __len__(self):
        return self._len

    @property
    def cached(self) -> bool:
        return self._cached


#### Array fn ####


def pad_examples(x: jax.Array, desired_batch_size: int) -> jax.Array:
    """Expand batch to desired size by repeating last slice."""
    batch_pad = desired_batch_size - x.shape[0]
    return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def per_host_sum_pmap(in_tree):
    """Execute psum on in_tree"s leaves over one device per host."""
    host2devices = collections.defaultdict(list)
    for d in jax.devices():
        host2devices[d.process_index].append(d)
    devices = [host2devices[k][0] for k in host2devices]
    host_psum = jax.pmap(lambda x: jax.lax.psum(x, "i"), "i", devices=devices)

    def pre_pmap(xs):
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)

    def post_pmap(xs):
        return jax.tree_util.tree_map(lambda x: x[0], xs)

    return post_pmap(host_psum(pre_pmap(in_tree)))


@jax.jit
def PPL(loss: jax.Array) -> jax.Array:
    """Compute perplexity from the cross-entropy loss"""
    return jnp.clip(jnp.exp(loss), a_max=1.0e5)
