"""
This file has been adapted from Ithaca: https://github.com/deepmind/ithaca/blob/main/ithaca/models/common_layers.py

Common layers used in models.
This implementation is from the Long Range Arena:
https://github.com/google-research/long-range-arena/tree/main/lra_benchmarks/models/bigbird
"""
from typing import Any, Callable, Iterable

from flax import linen as nn
from jax import lax
import jax.numpy as jnp
import numpy as np
import jax

PRNGKey = jax.Array
Shape = Iterable[int]
Dtype = jnp.dtype

ACTIVATION_FN_DICT = {
    "relu": nn.relu,
    "gelu": nn.gelu,
}


def grid_restack(all_vecs: Iterable[jax.Array]) -> jax.Array:
    """
    Grid restack for meta-performer.

    Given multiple sequences (lists) of batch x len x dim,
    reshape this such that all positions are side by side.
    for example (for illustrative purposes):
    inputs: [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
    outputs: [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

    Args:
        all_vecs: list of sequences of batch x len x dim

    Returns:
        Array of batch x (length x num_items) x dim.
    """
    cat_output = []
    for pos in range(all_vecs[0].shape[1]):
        pos_vecs = [x[:, None, pos, :] for x in all_vecs]
        cat_output += pos_vecs
    x2 = jnp.concatenate(cat_output, 1)
    return x2


def shift_right(x: jax.Array, axis: int = 1) -> jax.Array:
    """Shift the input to the right by padding on axis 1."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
    return padded[:, :-1]


def sinusoidal_init(
    max_len: int = 2048,
) -> Callable[[Any, Shape, jnp.dtype], jax.Array]:
    """
    1D Sinusoidal Position Embedding Initializer.

    Args:
        max_len: maximum possible length for the input

    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key: Any, shape: Shape, dtype: jnp.dtype = np.float32) -> jax.Array:
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_feature, 2) * -(np.log(10000.0) / d_feature))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    """
    Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
        posemb_init: positional embedding initializer, if None, then use a fixed
            (non-learned) sinusoidal embedding table.
        max_len: maximum possible length for the input.
        combine_type: how to add the embeddings ("concat" or "add")
    """

    posemb_init: Callable | None = None
    max_len: int = 512
    combine_type: str = "concat"

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        inputs_positions: jax.Array | None = None,
        cache: jax.Array | None = None,
    ) -> jax.Array:
        """Applies AddPositionEmbs module.
        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init.

        Args:
            inputs: input data.
            inputs_positions: input position indices for packed sequences.
            cache: flax attention cache for fast decoding.

        Returns:
            output: `(bs, timesteps, in_dim)`
        """
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3, but it is: %d" % inputs.ndim
        )
        batch_size = inputs.shape[0]
        length = inputs.shape[1]

        pos_emb_shape = (1, self.max_len, inputs.shape[-1])
        if self.posemb_init is None:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = sinusoidal_init(
                max_len=self.max_len,
            )(None, pos_emb_shape, None)
        else:
            pos_embedding = self.param("pos_embedding", self.posemb_init, pos_emb_shape)
        pe = pos_embedding[:, :length, :]
        # We abuse the same attention Cache mechanism to run positional embeddings
        # in fast predict mode. We could use state variables instead, but this
        # simplifies invocation with a single top-level cache context manager.
        # We only use the cache's position index for tracking decoding position.
        if cache:
            if self.is_initializing():
                cache.store(np.array((4, 1, 1), dtype=np.int32))
            else:
                cache_entry = cache.retrieve(None)
                i = cache_entry.i
                cache.store(cache_entry.replace(i=cache_entry.i + 1))
                _, _, df = pos_embedding.shape
                pe = lax.dynamic_slice(
                    pos_embedding, jnp.array((0, i, 0)), jnp.array((1, 1, df))
                )
        if inputs_positions is None:
            # normal unpacked case:
            if self.combine_type == "add":
                return inputs + pe
            elif self.combine_type == "concat":
                pe_broadcast = np.repeat(pe, batch_size, axis=0)
                return lax.concatenate([inputs, pe_broadcast], 2)
            else:
                raise ValueError("Wrong type value.")
        else:
            # for packed data we need to use known position indices:
            return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
    """
    Transformer MLP block.

    Attributes:
        mlp_dim: hidden size
        dtype: float data type
        out_dim: output dimension
        out_dropout: apply dropout on the output
        dropout_rate: dropout probability
        deterministic: if true it run in deterministic mode (no dropout)
        kernel_init: dense layer initializer
        bias_init: bias initializer
        activation_fn: activation function
    """

    mlp_dim: int
    dtype: Any = jnp.float32
    out_dim: int | None = None
    out_dropout: bool = True
    dropout_rate: float = 0.1
    deterministic: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    activation_fn: str = "gelu"

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = ACTIVATION_FN_DICT[self.activation_fn](x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
        output = nn.Dense(
            actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.out_dropout:
            output = nn.Dropout(rate=self.dropout_rate)(
                output, deterministic=self.deterministic
            )
        return output
