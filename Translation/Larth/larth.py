"""
Neural networks for translation.

This is based on Ithaca: https://github.com/deepmind/ithaca/blob/main/ithaca/models/model.py
and on the Jax MT example: https://github.com/google/flax/blob/main/examples/wmt/models.py 
"""
import flax.linen as nn
import jax
import jax.numpy as jnp

try:
    from bigbird import (
        TransformerEncoderDecoderBlock,
        LarthTranslationEncoder,
        LarthTranslationEncoderChar,
        LarthTranslationEncoderWord,
        LarthTranslationConfig,
    )
    from common_layers import (
        sinusoidal_init,
        AddPositionEmbs,
        shift_right,
    )
except ImportError:
    from .bigbird import (
        TransformerEncoderDecoderBlock,
        LarthTranslationEncoder,
        LarthTranslationEncoderChar,
        LarthTranslationEncoderWord,
        LarthTranslationConfig,
    )
    from .common_layers import (
        sinusoidal_init,
        AddPositionEmbs,
        shift_right,
    )


class LarthTranslationDecoder(nn.Module):
    """
    Larth decoder.

    TODO: move to bigbird.py

    Attributes:
        name
        config
    """

    name = "larth_decoder"
    config: LarthTranslationConfig

    @nn.compact
    def __call__(
        self,
        encoded: jax.Array,
        targets: jax.Array,
        decoder_mask: jax.Array,
        encoder_decoder_mask: jax.Array,
    ) -> jax.Array:
        """
        Translate the sequence.

        Args:
            encoded: encoded inputs with shape [batch, sequence, ...]
            targets: tokenized words with shape [batch, sequence]
            decoder_mask:
            encoder_decoder_mask:

        Returns:
            One-hot-encoding of the output sequence. Shape [batch, sequence, out vocabulary]
        """
        assert encoded.ndim == 3  # (batch, len, depth)
        assert targets.ndim == 2  # (batch, len)

        y = targets.astype(jnp.int32)

        if not self.config.decode:
            y = shift_right(y)

        # Target sequence embeddings
        y = nn.Embed(
            num_embeddings=self.config.out_word_vocab_size,
            features=self.config.emb_size,
            embedding_init=nn.initializers.normal(stddev=1.0),
        )(y)
        y = AddPositionEmbs(
            posemb_init=sinusoidal_init(max_len=self.config.max_len),
            max_len=self.config.max_len,
            combine_type="add",
            name="posembed_output",
        )(y)

        y = nn.Dropout(rate=self.config.dropout)(
            y, deterministic=self.config.deterministic
        )
        y = y.astype(self.config.dtype)
        for i in range(self.config.layers):
            y = TransformerEncoderDecoderBlock(
                config=self.config,
                name=f"encoderdecoderblock_{i}",
            )(
                y,
                encoded,
                decoder_mask=decoder_mask,
                encoder_decoder_mask=encoder_decoder_mask,
            )
        y = nn.LayerNorm(dtype=self.config.dtype, name="encoder_norm")(y)

        logits = nn.Dense(
            self.config.out_word_vocab_size,
            dtype=self.config.dtype,
            name="logitdense",
        )(y)

        return logits


class LarthTranslation(nn.Module):
    """
    Larth model.

    Atributes:
        config: configurations
    """

    config: LarthTranslationConfig

    def setup(self) -> None:
        if self.config.encoder_type == "char_word":
            self.encoder = LarthTranslationEncoder(config=self.config)
        elif self.config.encoder_type == "char":
            self.encoder = LarthTranslationEncoderChar(config=self.config)
        elif self.config.encoder_type == "word":
            self.encoder = LarthTranslationEncoderWord(config=self.config)

        self.decoder = LarthTranslationDecoder(config=self.config)

    def encode(self, chars: jax.Array, words: jax.Array) -> jax.Array:
        """
        Encode the input sequences.

        See: LarthTranslationEncoder

        Args:
            chars: tokenized characters. Shape [batch, sequence]
            words: tokenized words. Shape [batch, sequence]

        Returns:
            Encoded sequence with shape [batch, sequence, ...]
        """
        return self.encoder(chars, words)

    def decode(
        self, encoded: jax.Array, targets: jax.Array, mask: jax.Array
    ) -> jax.Array:
        """
        Decode the encoded sequences.

        See: LarthTranslationDecoder

        Args:
            encoded: encoded inputs with shape [batch, sequence, ...]
            targets: tokenized words with shape [batch, sequence]
            mask: mask marking the padded inputs

        Returns:
            One-hot-encoding of the output sequence. Shape [batch, sequence, out vocabulary]
        """
        if self.config.decode:
            # for fast autoregressive decoding only a special encoder-decoder mask is used
            decoder_mask = None
            # Shape [batch, 1, len, len)
            encoder_decoder_mask = nn.make_attention_mask(
                jnp.ones_like(targets) > 0, mask > 0, dtype=self.config.dtype
            )
        else:
            decoder_mask = nn.combine_masks(
                nn.make_attention_mask(
                    targets > 0, targets > 0, dtype=self.config.dtype
                ),
                nn.make_causal_mask(targets, dtype=self.config.dtype),
            )
            encoder_decoder_mask = nn.make_attention_mask(
                targets > 0, mask > 0, dtype=self.config.dtype
            )

        logits = self.decoder(
            encoded,
            targets,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
        )
        return logits.astype(self.config.dtype)

    def __call__(
        self, chars: jax.Array, words: jax.Array, targets: jax.Array
    ) -> jax.Array:
        """
        Translate Etruscan texts as character and word sequences.

        Args:
            chars: sequence of tokenized characters. Shape: [batch, sequence]
            words: sequence of tokenized words. Shape: [batch, sequence]
            targets: decoded sequence (use "<s>" for starting). Shape: [batch, decoded length]

        Returns:
            One-hot encoding in the output word vocab
        """
        encoded_torso = self.encode(chars, words)
        decoded = self.decode(encoded_torso, targets, chars)

        return decoded
