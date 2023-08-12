"""
This file is adapted from Ithaca: https://github.com/deepmind/ithaca/blob/main/ithaca/models/bigbird.py

It contains the model configuration dataclass, and the encoder and decoder classes.
"""

import abc

# from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

try:
    import bigbird_attention
    import common_layers
except ImportError:
    from . import bigbird_attention, common_layers


_DEFAULT_BLOCK_SIZE = 64
_DEFAULT_NUM_RAND_BLOCKS = 3

ENCODER_TYPES = ["char", "word", "char_word"]


@struct.dataclass
class LarthTranslationConfig:
    """
    Configuration for Larth

    Attributes:
        char_vocab_size: size of the char tokenizer vocabulary (input)
        word_char_emb_size: input embedding size
        word_vocab_size: size of the word tokenizer vocabulary (input)

        out_char_vocab_size: size of the char tokenizer vocabulary (output)
        out_word_vocab_size: size of the word tokenizer vocabulary (output)

        emb_size: embedding size
        max_len: maximum sequence length
        dropout: dropout probability
        dtype: float data type
        decode: run in decode mode (for inference)

        layers: number of attention blocks
        qkv_dim: size of the q, k, and v matrices
        mlp_dim: hidden size of the feedforwad part of the attention block
        num_heads: heads of the attention blocks
        attention_dropout: dropout probability
        activation_fn: in attention block
        block_size: BigBird block size
        num_rand_blocks: random blcoks in BigBird
        deterministic: run the model deterministically (for inference and testing)

        encoder_type: which encoder to use
    """

    # Char and word input
    char_vocab_size: int = 164
    word_char_emb_size: int = 512
    word_vocab_size: int = 10000

    # Output
    out_char_vocab_size: int = 164
    out_word_vocab_size: int = 5000

    #
    emb_size: int = 512
    max_len: int = 1024
    dropout: float = 0.1
    dtype: jnp.dtype = "float32"
    decode: bool = False

    # Attention block
    layers: int = 1
    qkv_dim: int = 512
    mlp_dim: int = 1024
    num_heads: int = 8
    attention_dropout: float = 0.1
    activation_fn: str = "gelu"
    block_size: int = _DEFAULT_BLOCK_SIZE
    num_rand_blocks: int = _DEFAULT_NUM_RAND_BLOCKS
    deterministic: bool = True

    encoder_type: str = "char_word"

    def __post_init__(self):
        if self.encoder_type not in ENCODER_TYPES:
            raise ValueError(
                f"encoder_type must be in {ENCODER_TYPES}, but {self.encoder_type} was found"
            )


class BigBirdEncoderBlock(nn.Module):
    """
    BigBird layer (https://arxiv.org/abs/2007.14062).
    Adapted from Ithaca: https://github.com/deepmind/ithaca/blob/main/ithaca/models/bigbird.py

    Attributes (from config):
        qkv_dim: dimension of the query/key/value
        mlp_dim: dimension of the mlp on top of attention block
        num_heads: number of heads
        dtype: the dtype of the computation (default: float32).
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate for attention weights
        deterministic: bool, deterministic or not (to apply dropout)
        activation_fn: Activation function ("relu", "gelu")
        block_size: Size of attention blocks.
        num_rand_blocks: Number of random blocks.
        connectivity_seed: Optional seed for random block sparse attention.
    """

    config: LarthTranslationConfig
    connectivity_seed: int | None = None

    @nn.compact
    def __call__(self, inputs: jax.Array, padding_mask: jax.Array) -> jax.Array:
        """
        Applies BigBirdBlock module.

        Args:
            inputs: input data
            inputs_segmentation: input segmentation info for packed examples.
            padding_mask: bool, mask padding tokens, [b, l, 1]

        Returns:
            output after transformer block.
        """

        # Attention block.
        assert inputs.ndim == 3
        # x = common_layers.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.LayerNorm(dtype=self.config.dtype)(inputs)
        x = bigbird_attention.BigBirdSelfAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            qkv_features=self.config.qkv_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.config.attention_dropout,
            deterministic=self.config.deterministic,
            block_size=self.config.block_size,
            num_rand_blocks=self.config.num_rand_blocks,
            connectivity_seed=self.connectivity_seed,
        )(
            x,
            padding_mask=padding_mask,
        )

        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )
        x = x + inputs
        # MLP block.
        y = nn.LayerNorm(dtype=self.config.dtype)(x)
        y = common_layers.MlpBlock(
            mlp_dim=self.config.mlp_dim,
            dtype=self.config.dtype,
            dropout_rate=self.config.dropout,
            deterministic=self.config.deterministic,
            activation_fn=self.config.activation_fn,
        )(y)

        return x + y


class TransformerEncoderDecoderBlock(nn.Module):
    """
    Transformer encoder-decoder layer.
    Adapted from Ithaca: https://github.com/deepmind/ithaca/blob/main/ithaca/models/bigbird.py

    Attributes (from config):
        qkv_dim: dimension of the query/key/value
        mlp_dim: dimension of the mlp on top of attention block
        num_heads: number of heads
        dtype: the dtype of the computation (default: float32).
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate for attention weights
        deterministic: bool, deterministic or not (to apply dropout)
        activation_fn: Activation function ("relu", "gelu")
        decoder: whether to run it in decode mode
    """

    config: LarthTranslationConfig

    @nn.compact
    def __call__(
        self,
        targets: jax.Array,
        encoded: jax.Array,
        decoder_mask: jax.Array,
        encoder_decoder_mask: jax.Array,
    ) -> jax.Array:
        """
        Applies TransformerEncoderDecoderBlock module.

        Args:
            targets: input data for decoder
            encoded: input data from encoder
            decoder_mask: decoder self-attention mask.
            encoder_decoder_mask: encoder-decoder attention mask.

        Returns:
            output after transformer encoder-decoder block.
        """
        assert targets.ndim == 3

        x = nn.LayerNorm(dtype=self.config.dtype)(targets)
        x = nn.SelfAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            qkv_features=self.config.qkv_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.config.attention_dropout,
            deterministic=self.config.deterministic,
            decode=self.config.decode,
        )(x, decoder_mask)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )
        x = x + targets

        # Encoder-Decoder block.
        y = nn.LayerNorm(dtype=self.config.dtype)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            qkv_features=self.config.qkv_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.config.attention_dropout,
            deterministic=self.config.deterministic,
        )(y, encoded, encoder_decoder_mask)

        y = nn.Dropout(rate=self.config.dropout)(
            y, deterministic=self.config.deterministic
        )
        y = y + x

        # MLP block.
        z = nn.LayerNorm(dtype=self.config.dtype)(y)
        y = common_layers.MlpBlock(
            mlp_dim=self.config.mlp_dim,
            dtype=self.config.dtype,
            dropout_rate=self.config.dropout,
            deterministic=self.config.deterministic,
            activation_fn=self.config.activation_fn,
        )(y)
        return y + z


class LarthTranslationEncoderBase(abc.ABC, nn.Module):
    """
    Base class for Larth encoders.
    This handles the positional embedding and attention part.

    The input embedding is done by the subclasses

    Attributes:
        name: name of the module
        config: module parameters
    """

    name: str = "larth_encoder_base"
    config: LarthTranslationConfig

    def _encode(self, x: jax.Array, padding_mask: jax.Array) -> jax.Array:
        """
        Apply attention on the input
        """
        # Positional embeddings
        pe_init = common_layers.sinusoidal_init(max_len=self.config.max_len)
        x = common_layers.AddPositionEmbs(
            posemb_init=pe_init,
            max_len=self.config.max_len,
            combine_type="add",
            name="posembed_input",
        )(x)

        x = nn.Dropout(rate=self.config.dropout, name="dropout_0")(
            x, self.config.deterministic
        )

        for i in range(self.config.layers):
            x = BigBirdEncoderBlock(
                config=self.config,
                connectivity_seed=i,  # rng seed: block counter,
                name=f"encoder_block_{i}",
            )(x, padding_mask)

        x = nn.LayerNorm(dtype=self.config.dtype, name="encoder_norm")(x)
        return x

    @abc.abstractmethod
    def __call__(self, chars: jax.Array, words: jax.Array) -> jax.Array:
        pass


class LarthTranslationEncoder(LarthTranslationEncoderBase):
    """
    Larth encoder that uses both the character and word sequences.

    Attributes:
        name: name of the module
        config: module parameters
    """

    name = "larth_encoder"

    def setup(self) -> None:
        self._char_emb = nn.Embed(
            num_embeddings=self.config.char_vocab_size,
            features=self.config.word_char_emb_size,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="char_embeddings",
        )

        self._word_emb = nn.Embed(
            num_embeddings=self.config.word_vocab_size,
            features=self.config.word_char_emb_size,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="word_embeddings",
        )

    @nn.compact
    def __call__(self, chars: jax.Array, words: jax.Array) -> jax.Array:
        """
        Encode the input sequences.
        Chars and words must have the same shape.
        The padding is assumed to be as follow:
        Position: 0    1    2    3    4 5  6  7 8  9  10 11      12      13      ...
        Chars:    T    h    i    s    _ i  s  _ a  n  _  e       x       a       ...
        Words:    This This This This _ is is _ an an _  example example example ...

        Args:
            chars: tokenized characters. Shape [batch, sequence]
            words: tokenized words. Shape [batch, sequence]

        Returns:
            Encoded sequence with shape [batch, sequence, ...]
        """
        # Embeddings and padding mask
        char_emb = self._char_emb(chars)
        word_emb = self._word_emb(words)
        # Concat embeddings
        x = jnp.concatenate([char_emb, word_emb], 2)

        padding = jnp.where(chars > 0, 1, 0)
        padding_mask = padding[..., jnp.newaxis]  # from (len,) to (len, 1)

        return super()._encode(x, padding_mask)


class LarthTranslationEncoderChar(LarthTranslationEncoderBase):
    """
    Larth encoder that uses only the character sequence.

    Attributes:
        name: name of the module
        config: module parameters
    """

    name = "larth_encoder_char"

    @nn.compact
    def __call__(self, chars: jax.Array, words: jax.Array | None = None) -> jax.Array:
        """
        Encode the char sequence.

        Args:
            chars: tokenized characters. Shape [batch, sequence]
            words: ignored

        Returns:
            Encoded sequence with shape [batch, sequence, ...]
        """
        # Embeddings and padding mask
        x = nn.Embed(
            num_embeddings=self.config.char_vocab_size,
            features=self.config.word_char_emb_size,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="char_embeddings",
        )(chars)

        padding = jnp.where(chars > 0, 1, 0)
        padding_mask = padding[..., jnp.newaxis]  # from (len,) to (len, 1)

        return super()._encode(x, padding_mask)


class LarthTranslationEncoderWord(LarthTranslationEncoderBase):
    """
    Larth encoder that uses only the word sequence.

    Attributes:
        name: name of the module
        config: module parameters
    """

    name = "larth_encoder_word"

    @nn.compact
    def __call__(self, chars: jax.Array | None, words: jax.Array) -> jax.Array:
        """
        Encode the word sequence.

        Args:
            chars: ignored
            words: tokenized words. Shape [batch, sequence]

        Returns:
            Encoded sequence with shape [batch, sequence, ...]
        """
        # Embeddings and padding mask
        x = nn.Embed(
            num_embeddings=self.config.word_vocab_size,
            features=self.config.word_char_emb_size,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="word_embeddings",
        )(words)

        padding = jnp.where(words > 0, 1, 0)
        padding_mask = padding[..., jnp.newaxis]  # from (len,) to (len, 1)

        return super()._encode(x, padding_mask)
