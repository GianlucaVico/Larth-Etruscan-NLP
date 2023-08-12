"""
General functions used for training
"""
import json
import logging
from typing import Dict, Iterator, List, Tuple, Self, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow as pa
import yaml
from flax.struct import dataclass
from flax.training import common_utils


##### Config #####
def parse_config(path: str) -> Dict:
    """
    Load a config file (json or yaml)

    Args:
        path: file

    Returns:
        dictionary with configuration
    """
    with open(path) as f:
        if path.endswith("json"):
            d = json.load(f)
        elif path.endswith("yaml") or path.endswith("yml"):
            d = yaml.safe_load(f)
        else:
            raise Exception("Unsupported config file")
    return d


def save_config(path: str, d: Dict) -> None:
    """
    Save a config file

    Args:
        path: where to store the configuration
        d: dictionary with configuration
    """
    with open(path, "w") as f:
        if path.endswith("json"):
            json.dump(d, f)
        elif path.endswith("yaml") or path.endswith("yml"):
            yaml.safe_dump(d, f)
        else:
            raise Exception("Unsupported config file")


LANGUAGES = ["english", "etruscan", "greek", "latin"]
MODES = ["translation", "restoration", "lm"]  # NOT USED
DATASET_TYPES = ["csv", "tatoeba"]
ALIGNMENTS = ["same", "space"]


@dataclass
class TrainConfig:
    """
    All the paramenters for training and loading the dataset
    """

    # Size of the training batches. Eval batches have half the size
    batch_size: int = 16
    # Learning rate
    lr: float = 5e-3

    warmup_steps: int = 1

    # Adam weight decay
    weight_decay: float = 5e-4

    # Where to save the results
    workdir: str = "log/"

    label_smoothing: float = 0

    # Whether to resume a training
    restore_checkpoints: bool = False

    # Decoding beam size
    beam_size: int = 1
    length_penalty: float = 0.6

    # Number of training epochs
    epochs: int = 1

    # Evaluate the model after this many epochs
    eval_every_epochs: int = 1

    # Save the model after this many epochs
    checkpoint_every_epochs: int = 1

    # Which model to load
    restore_from: str | None = None

    # Pre-load the entire dataset
    cached: bool = False

    # CSV with the dataset
    dataset_type: str = "csv"  # csv or tatoeba
    subset: str | None = (
        None  # etp, ciep, both for Etruscan; dev for tatoeba; None for unset
    )
    data_path: str = "../../Data/Etruscan.csv"
    etruscan_vocab: str | None = "../../Data/EPT_POS.csv"

    # Name of the source and target tokenizer (_word and _char are added automatically)
    source_model: str = "../../Data/etruscan"
    target_model: str = "../../Data/english"
    alignment: str = "same"

    # Portion of data used for training
    train_size: float = 0.9

    # RNG seed
    seed: int = 0

    # Load only two batches
    debug: bool = False

    # Languages
    source_lang: str = "etruscan"
    target_lang: str = "english"

    # Whether to train or eval
    train: bool = True
    eval: bool = True
    # Translation / Restoration / LM
    mode: str = "translation"  # NOT USED

    # Number of new examples generated
    name_augmentation_max_replacements: int = 0

    # Probabily of corrupting an end of a token
    unk_augmentation_prob: float = 0.0
    # Average number of characters currupted (when corruption happens)
    unk_augmentation_len: float = 1.0
    # Each iteration generate a new example for each given example
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
        if self.source_lang not in LANGUAGES:
            raise ValueError(
                f"Source language must be any of {LANGUAGES}, but {self.source_lang} found"
            )
        if self.target_lang not in LANGUAGES:
            raise ValueError(
                f"Target language must be any of {LANGUAGES}, but {self.target_lang} found"
            )

        if self.mode not in MODES:
            raise ValueError(f"Mode must be any of {MODES}, but {self.mode} found")

        if self.dataset_type not in DATASET_TYPES:
            raise ValueError(
                f"Dataset type must be any of {DATASET_TYPES}, but {self.dataset_type} found"
            )

        if self.alignment not in ALIGNMENTS:
            raise ValueError(
                f"Dataset type must be any of {ALIGNMENTS}, but {self.alignment} found"
            )

        # Check language pairs.
        # Both directions are fine. Only one language is fine
        # TODO: English -> english is ambiguous
        pairs = [{"etruscan", "english"}, {"greek", "english"}, {"latin", "english"}]
        pair = set((self.source_lang, self.target_lang))
        if not (len(pair) == 1 or pair in pairs):
            raise ValueError(
                f"Invalid language pair. Found {pair}. The pairs supported are {pairs}"
            )


###### Data #####


class DataLoader:
    """
    Create the batched and iterate through the datasets

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
        self.ds: List[pa.RecordBatch] | List[
            Dict[str, jax.Array | str]
        ] = ds.to_batches(batch_size)
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

    def __iter__(self) -> Self:
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
        source_chars, source_words = self.pad_pairs(
            d["source_chars"], d["source_words"]
        )
        target_chars, target_words = self.pad_pairs(
            d["target_chars"], d["target_words"]
        )
        source_chars = jnp.array(source_chars)[:, : self.max_len]
        source_words = jnp.array(source_words)[:, : self.max_len]
        target_chars = jnp.array(target_chars)[:, : self.max_len]
        target_words = jnp.array(target_words)[:, : self.max_len]

        if self.array_only:
            return {
                "source_chars": source_chars,
                "source_words": source_words,
                "target_chars": target_chars,
                "target_words": target_words,
            }
        else:
            return {
                "source": [
                    i[: self.max_len - 2] for i in d["source"]
                ],  # bos and eos tokens
                "source_chars": source_chars,
                "source_words": source_words,
                "target": [i[: self.max_len - 2] for i in d["target"]],
                "target_chars": target_chars,
                "target_words": target_words,
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
        new_l = self._pad(l, l_lens, self.max_len)
        return new_l

    def pad_pairs(
        self, l: List[List[int]], o: List[List[int]]
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Pad pairs of sequences to the same length.

        Args:
            l: first list of sequences
            o: other list of sequences

        Returns:
            Tuple with two jax arrays
        """
        l_lens, _ = self._get_lens(l)
        o_lens, _ = self._get_lens(o)
        new_l = self._pad(l, l_lens, self.max_len)
        new_o = self._pad(o, o_lens, self.max_len)
        return new_l, new_o

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


##### Learning rate #####


def rsqrt_schedule(
    init_value: float,
    shift: int = 0,
) -> Callable[[int], float]:
    """Applies a reverse square-root schedule.

    The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

    Args:
        init_value: Base learning rate (before applying the rsqrt schedule).
        shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
        schedule makes it less steep in the beginning (close to 0).

    Returns:
        A schedule `count -> learning_rate`.
    """

    def schedule(count):
        return init_value * (count + shift) ** -0.5 * shift**0.5

    return schedule


def create_learning_rate_schedule(
    lr: float, warmup_steps: int
) -> Callable[[int], float]:
    """
    Args:
        lr: learning rate
        warmup_steps: initial steps where the linear schedule is used

    Returns:
        Schedule as function epoch->learning rate
    """
    return optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0, end_value=lr, transition_steps=warmup_steps
            ),
            rsqrt_schedule(init_value=lr, shift=warmup_steps),
        ],
        boundaries=[warmup_steps],
    )


##### Metrics #####


def compute_weighted_cross_entropy(
    logits: jax.Array,
    targets: jax.Array,
    weights: jax.Array | None = None,
    label_smoothing: float = 0.0,
) -> Tuple[float, float]:
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
        logits: [batch, length, num_classes] float array.
        targets: categorical targets [batch, length] int array.
        weights: None or array of shape [batch, length].
        label_smoothing: label smoothing constant, used to determine the on and off values.

    Returns:
        Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            f"Incorrect shapes. Got shape {str(logits.shape)} logits and {str(targets.shape)} targets"
        )
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence)
        + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_targets = common_utils.onehot(
        targets, vocab_size, on_value=confidence, off_value=low_confidence
    )

    loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
    loss = loss - normalizing_constant

    normalizing_factor = np.prod(targets.shape)

    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum(), normalizing_factor


def compute_weighted_accuracy(
    logits: jax.Array, targets: jax.Array, weights: jax.Array | None = None
) -> Tuple[float, float]:
    """Compute weighted accuracy for log probs and targets.

    Args:
        logits: [batch, length, num_classes] float array.
        targets: categorical targets [batch, length] int array.
        weights: None or array of shape [batch, length]

    Returns:
        Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            f"Incorrect shapes. Got shape {str(logits.shape)} logits and {str(targets.shape)} targets"
        )
    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    normalizing_factor = np.prod(logits.shape[:-1])
    if weights is not None:
        loss = loss * weights
        normalizing_factor = weights.sum()

    return loss.sum(), normalizing_factor


def compute_metrics(
    logits: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    """
    Compute summary metrics.

    Args:
        logits: [batch, length, num_classes] float array.
        labels: categorical targets [batch, length] int array.
        weights: None or array of shape [batch, length].
        label_smoothing: label smoothing constant, used to determine the on and off values.

    Returns:
        Dictionary with "loss", "accuracy", "denominator"
    """
    loss, weight_sum = compute_weighted_cross_entropy(
        logits, labels, weights, label_smoothing
    )
    acc, _ = compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        "loss": loss,
        "accuracy": acc,
        "denominator": weight_sum,
    }
    metrics = jax.lax.psum(metrics, axis_name="batch")
    return metrics


#### Array fn ####


def pad_examples(x: jax.Array, desired_batch_size: int) -> jax.Array:
    """Expand batch to desired size by repeating last slice."""
    batch_pad = desired_batch_size - x.shape[0]
    return np.concatenate([x, jnp.tile(x[-1], (batch_pad, 1))], axis=0)


def tohost(x: jax.Array) -> np.ndarray:
    """Collect batches from all devices to host and flatten batch dimensions."""
    n_device, n_batch, *remaining_dims = x.shape
    return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))
