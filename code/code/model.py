"""
model.py
========
ConvLSTM-AE architecture (Section 5.1, Fig. 3) and the Critic network
used during adversarial pretraining (Section 5.2, Fig. 4).

Architecture (from Fig. 3):
  Encoder
  -------
  ConvLSTM2D(16 filters, kernel=(1,7), padding='same', return_sequences=True, activation='tanh')
  BatchNormalization
  Dropout(0.2)
  ConvLSTM2D(32 filters, kernel=(1,7), padding='same', return_sequences=False, activation='tanh')
  BatchNormalization
  Flatten
  Dense(latent_dim)          ← Latent space Z

  Decoder
  -------
  Dense → Reshape
  ConvLSTM2D(32 filters, kernel=(1,7), padding='same', return_sequences=True,  activation='tanh')
  BatchNormalization
  Dropout(0.2)
  ConvLSTM2D(16 filters, kernel=(1,7), padding='same', return_sequences=True,  activation='tanh')
  TimeDistributed(Dense(n_features, activation='relu'))  ← Reconstruction

  Critic (for ACAI pretraining, Section 5.2)
  -------
  Mirrors the encoder; final Dense(1) predicts interpolation coefficient α.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder(n_steps: int,
                  n_length: int,
                  n_features: int = 1,
                  latent_dim: int = 10,
                  dropout_rate: float = 0.2) -> Model:
    """
    Input shape : (batch, n_steps, 1, n_length, n_features)
    Output shape: (batch, latent_dim)
    """
    inp = layers.Input(shape=(n_steps, 1, n_length, n_features), name="encoder_input")

    # Block 1
    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(1, 7),
        padding="same",
        return_sequences=True,
        activation="tanh",
        name="convlstm_enc_1",
    )(inp)
    x = layers.BatchNormalization(name="bn_enc_1")(x)
    x = layers.Dropout(dropout_rate, name="drop_enc_1")(x)

    # Block 2
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(1, 7),
        padding="same",
        return_sequences=False,
        activation="tanh",
        name="convlstm_enc_2",
    )(x)
    x = layers.BatchNormalization(name="bn_enc_2")(x)

    # Flatten → Dense latent projection
    x = layers.Flatten(name="flatten_enc")(x)
    z = layers.Dense(latent_dim, name="latent_space")(x)

    return Model(inp, z, name="Encoder")


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def build_decoder(n_steps: int,
                  n_length: int,
                  n_features: int = 1,
                  latent_dim: int = 10,
                  dropout_rate: float = 0.2) -> Model:
    """
    Input shape : (batch, latent_dim)
    Output shape: (batch, n_steps, 1, n_length, n_features)
    """
    # The intermediate spatial volume after the encoder flatten
    # encoder block 2 output: (1, n_length, 32)  → flattened = n_length * 32
    intermediate_dim = 1 * n_length * 32

    inp = layers.Input(shape=(latent_dim,), name="decoder_input")

    x = layers.Dense(intermediate_dim, name="dense_dec")(inp)
    x = layers.Reshape((1, 1, n_length, 32), name="reshape_dec")(x)

    # Repeat along the time axis to reconstruct n_steps frames
    x = layers.Lambda(
        lambda t: tf.repeat(t, repeats=n_steps, axis=1),
        name="repeat_steps",
    )(x)

    # Block 1
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(1, 7),
        padding="same",
        return_sequences=True,
        activation="tanh",
        name="convlstm_dec_1",
    )(x)
    x = layers.BatchNormalization(name="bn_dec_1")(x)
    x = layers.Dropout(dropout_rate, name="drop_dec_1")(x)

    # Block 2
    x = layers.ConvLSTM2D(
        filters=16,
        kernel_size=(1, 7),
        padding="same",
        return_sequences=True,
        activation="tanh",
        name="convlstm_dec_2",
    )(x)

    # Final reconstruction via TimeDistributed Dense
    out = layers.TimeDistributed(
        layers.Dense(n_features, activation="relu"),
        name="output_reconstruction",
    )(x)

    return Model(inp, out, name="Decoder")


# ---------------------------------------------------------------------------
# Critic network (for ACAI adversarial regularisation)
# ---------------------------------------------------------------------------

def build_critic(n_steps: int,
                 n_length: int,
                 n_features: int = 1,
                 dropout_rate: float = 0.2) -> Model:
    """
    Predicts interpolation coefficient α from a reconstructed sample x̂_α.

    Architecture: Flatten → Dense → Dense → Dense(1, sigmoid)

    WHY NOT ConvLSTM:
    Using a ConvLSTM critic caused TensorFlow graph cycle errors and
    latent space explosion when trained alongside a ConvLSTM encoder/decoder.
    The critic only needs to predict a scalar from a sequence — it does not
    need temporal modelling. A simple Dense network is stable, fast, and
    sufficient for the ACAI discriminator task.

    Input shape : (batch, n_steps, 1, n_length, n_features)
    Output shape: (batch, 1)
    """
    inp = layers.Input(shape=(n_steps, 1, n_length, n_features),
                       name="critic_input")

    # Flatten the full sequence to a 1-D vector
    x = layers.Flatten(name="critic_flatten")(inp)

    # Three Dense layers with decreasing size
    x = layers.Dense(256, activation="relu", name="critic_dense_1")(x)
    x = layers.Dropout(dropout_rate, name="critic_drop_1")(x)

    x = layers.Dense(128, activation="relu", name="critic_dense_2")(x)
    x = layers.Dropout(dropout_rate, name="critic_drop_2")(x)

    x = layers.Dense(64,  activation="relu", name="critic_dense_3")(x)

    # Scalar output — predicts interpolation coefficient α
    # Linear activation: sigmoid causes gradient saturation and Z_var explosion
    out = layers.Dense(1, activation="linear", name="critic_output")(x)

    return Model(inp, out, name="Critic")


# ---------------------------------------------------------------------------
# Convenience: full autoencoder (encoder + decoder composed)
# ---------------------------------------------------------------------------

def build_autoencoder(n_steps: int,
                      n_length: int,
                      n_features: int = 1,
                      latent_dim: int = 10,
                      dropout_rate: float = 0.2):
    """
    Returns (encoder, decoder, autoencoder) as a named tuple.
    The autoencoder maps input → reconstructed output directly.
    """
    encoder = build_encoder(n_steps, n_length, n_features, latent_dim, dropout_rate)
    decoder = build_decoder(n_steps, n_length, n_features, latent_dim, dropout_rate)

    inp   = layers.Input(shape=(n_steps, 1, n_length, n_features), name="ae_input")
    z     = encoder(inp)
    x_hat = decoder(z)
    autoencoder = Model(inp, x_hat, name="ConvLSTM_AE")

    return encoder, decoder, autoencoder


# ---------------------------------------------------------------------------
# Quick summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N_STEPS, N_LEN, N_FEAT, LAT = 7, 48, 1, 10

    enc, dec, ae = build_autoencoder(N_STEPS, N_LEN, N_FEAT, LAT)
    critic       = build_critic(N_STEPS, N_LEN, N_FEAT)

    print("\n=== Encoder ===")
    enc.summary()
    print("\n=== Decoder ===")
    dec.summary()
    print("\n=== Critic ===")
    critic.summary()
    print("\n=== Autoencoder ===")
    ae.summary()
