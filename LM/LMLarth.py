"""
Transformer language model.

Based on Flax lm example and Larth.
"""
import sys

sys.path.append("../")

from Translation import Larth
import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class LarthLMConfig:
    """
    Model hyperparameters

    Attributes:
        vocab_size: number of words
        emb_size: embedding size
        max_len: maximum sequence lenth
        dropout: dropout probability
        dtype: float data type
        decode: run the model in decode mode

        layers: number of attention blocks
        qkv_dim: dimension of the Q, K and V matrices
        mlp_dim: dimension of the feed forwards layers
        num_heads: number of attention heads
        attention_dropout: dropout probability in the attention blocks
        activation_fn: for the feed forward layers
        deterministic: run the model in deterministic mode (i.e. without dropout)
    """

    vocab_size: int = 10000

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
    deterministic: bool = True


class TransformerEncoderDecoder(nn.Module):
    """
    Transformer encoder-decoder layer.

    Args:
      config: LarthLMConfig dataclass containing hyperparameters.
    """

    config: LarthLMConfig

    @nn.compact
    def __call__(self, inputs: jax.Array, decoder_mask: jax.Array = None) -> jax.Array:
        """
        Applies EncoderDecoder1DBlock module.

        Args:
          inputs: input data for decoder
          decoder_mask: decoder self-attention mask.

        Returns:
          output after transformer encoder-decoder block.
        """
        # Decoder block.
        assert inputs.ndim == 3

        x = nn.LayerNorm(dtype=self.config.dtype)(inputs)
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
        x = x + inputs

        # MLP block.
        z = nn.LayerNorm(dtype=self.config.dtype)(x)
        z = Larth.common_layers.MlpBlock(
            mlp_dim=self.config.mlp_dim,
            dtype=self.config.dtype,
            dropout_rate=self.config.dropout,
            deterministic=self.config.deterministic,
            activation_fn=self.config.activation_fn,
        )(z)
        return x + z


class Decoder(nn.Module):
    """
    Transformer Model Decoder for LM.

    Args:
      config: LarthLMConfig dataclass containing hyperparameters.
    """

    config: LarthLMConfig

    @nn.compact
    def __call__(self, inputs: jax.Array, decoder_mask: jax.Array = None) -> jax.Array:
        """
        Applies Transformer model on the inputs.

        Args:
          encoded: encoded input data from encoder.
          inputs: input data.
          decoder_mask: decoder self-attention mask.

        Returns:
          output of a transformer decoder.
        """
        assert inputs.ndim == 2  # (batch, len)

        # Target Embedding
        output_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.emb_size,
            embedding_init=nn.initializers.normal(stddev=1.0),
        )

        y = inputs.astype("int32")
        if not self.config.decode:
            y = Larth.shift_right(y)
        y = output_embed(y)
        y = Larth.AddPositionEmbs(
            posemb_init=Larth.sinusoidal_init(max_len=self.config.max_len),
            max_len=self.config.max_len,
            combine_type="add",
            name="posembed",
        )(y)
        y = nn.Dropout(rate=self.config.dropout)(
            y, deterministic=self.config.deterministic
        )

        y = y.astype(self.config.dtype)

        # Target-Input Decoder
        for lyr in range(self.config.layers):
            y = TransformerEncoderDecoder(
                config=self.config, name=f"encoderdecoderblock_{lyr}"
            )(y, decoder_mask=decoder_mask)

        y = nn.LayerNorm(dtype=self.config.dtype, name="encoderdecoder_norm")(y)

        logits = nn.Dense(
            self.config.vocab_size,
            dtype=self.config.dtype,
            name="logitdense",
        )(y)

        return logits


class LarthLM(nn.Module):
    """
    Transformer pure decoder stack for language modelling.

    Args:
        config: LarthLMConfig dataclass containing hyperparameters.
    """

    config: LarthLMConfig

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        Applies LarthLM on the inputs.

        Args:
            inputs: target data.

        Returns:
            logits array from transformer decoder.
        """
        # Make padding attention masks.
        if self.config.decode:
            # for fast autoregressive decoding we use no decoder mask
            decoder_mask = None
        else:
            decoder_mask = nn.combine_masks(
                nn.make_attention_mask(inputs > 0, inputs > 0, dtype=self.config.dtype),
                nn.make_causal_mask(inputs, dtype=self.config.dtype),
            )

        # Add segmentation block-diagonal attention masks if using segmented data.
        logits = Decoder(config=self.config, name="decoder")(
            inputs, decoder_mask=decoder_mask
        )
        return logits.astype(self.config.dtype)
