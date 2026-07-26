"""
baselines.py
============
Baseline models referenced in Section 6.1 and Tables 2/3.

Deep baselines:
  - DEC   (Deep Embedded Clustering,          Xie et al. 2016 [33])
  - IDEC  (Improved DEC,                      Guo   et al. 2017 [7])
  - DynAE (Dynamic Autoencoder,               Mrabah et al. 2019 [15])
  - LSTM-AE (LSTM Autoencoder,                Yang   et al. 2022 [35])

Classical baselines:
  - K-Means, Agglomerative, DBSCAN, Spectral, GMM
    All from sklearn; same flat feature space as the raw/flattened input.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.cluster import (KMeans, AgglomerativeClustering,
                              DBSCAN, SpectralClustering)
from sklearn.mixture  import GaussianMixture
from sklearn.metrics  import (silhouette_score, davies_bouldin_score,
                               calinski_harabasz_score)


# ===========================================================================
# ─── DEEP BASELINES ──────────────────────────────────────────────────────── #
# ===========================================================================


# ---------------------------------------------------------------------------
# LSTM-AE  (Yang et al. 2022 [35])
# ---------------------------------------------------------------------------

def build_lstm_ae(seq_len:    int,
                  n_features: int = 1,
                  latent_dim: int = 32,
                  dropout:    float = 0.2):
    """
    Simple bidirectional LSTM autoencoder for temporal load profiling.
    Input shape: (batch, seq_len, n_features)
    """
    inp = layers.Input(shape=(seq_len, n_features), name="lstm_ae_input")

    # Encoder
    x = layers.LSTM(64, return_sequences=True, name="lstm_enc_1")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(32, return_sequences=False, name="lstm_enc_2")(x)
    z = layers.Dense(latent_dim, name="lstm_latent")(x)

    # Decoder — repeat latent to reconstruct seq_len steps
    x = layers.RepeatVector(seq_len, name="repeat")(z)
    x = layers.LSTM(32, return_sequences=True, name="lstm_dec_1")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(64, return_sequences=True, name="lstm_dec_2")(x)
    out = layers.TimeDistributed(layers.Dense(n_features, activation="relu"),
                                  name="lstm_output")(x)

    encoder    = Model(inp, z,   name="LSTM_Encoder")
    autoencoder = Model(inp, out, name="LSTM_AE")
    return encoder, autoencoder


def run_lstm_ae(X_flat:    np.ndarray,
                K:         int,
                latent_dim: int   = 32,
                epochs:    int   = 100,
                batch_size: int  = 256,
                lr:        float = 1e-3) -> np.ndarray:
    """
    Train LSTM-AE, encode X, then K-means in latent space.
    X_flat: (N, T)  — households × timestamps (2D, will be reshaped).
    Returns cluster labels (N,).
    """
    N, T = X_flat.shape
    X3 = X_flat.reshape(N, T, 1).astype(np.float32)

    encoder, ae = build_lstm_ae(T, n_features=1, latent_dim=latent_dim)
    ae.compile(optimizer=optimizers.Adam(lr), loss="mse")
    ae.fit(X3, X3, epochs=epochs, batch_size=batch_size, verbose=0)

    Z      = encoder.predict(X3, batch_size=batch_size, verbose=0)
    km     = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(Z)
    return labels


# ---------------------------------------------------------------------------
# Standard flat autoencoder (shared by DEC, IDEC, DynAE baselines)
# ---------------------------------------------------------------------------

def build_flat_ae(input_dim:  int,
                  latent_dim: int = 10,
                  hidden_dims: list = None,
                  dropout:    float = 0.2):
    """
    Dense autoencoder.  Architecture from [5,7,15]:
    Input → Dense(500) → Dense(500) → Dense(2000) → Dense(latent_dim)
    """
    if hidden_dims is None:
        hidden_dims = [500, 500, 2000]

    # Encoder
    enc_inp = layers.Input(shape=(input_dim,), name="flat_enc_input")
    x = enc_inp
    for i, h in enumerate(hidden_dims):
        x = layers.Dense(h, activation="relu", name=f"enc_{i}")(x)
        x = layers.Dropout(dropout)(x)
    z = layers.Dense(latent_dim, name="flat_latent")(x)
    encoder = Model(enc_inp, z, name="FlatEncoder")

    # Decoder
    dec_inp = layers.Input(shape=(latent_dim,), name="flat_dec_input")
    x = dec_inp
    for i, h in enumerate(reversed(hidden_dims)):
        x = layers.Dense(h, activation="relu", name=f"dec_{i}")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(input_dim, activation="sigmoid", name="flat_output")(x)
    decoder = Model(dec_inp, out, name="FlatDecoder")

    return encoder, decoder


# ---------------------------------------------------------------------------
# DEC  (Xie et al. 2016 [33])
# ---------------------------------------------------------------------------

class DEC:
    """
    Deep Embedded Clustering.
    Pretrain AE with MSE, then fine-tune using KL-divergence
    between soft assignments Q and auxiliary target distribution P.
    """

    def __init__(self, input_dim: int, K: int, latent_dim: int = 10,
                 alpha: float = 1.0):
        self.K         = K
        self.alpha     = alpha
        self.encoder, self.decoder = build_flat_ae(input_dim, latent_dim)
        self.centroids = None

        # Full AE
        inp   = layers.Input(shape=(input_dim,))
        z     = self.encoder(inp)
        x_hat = self.decoder(z)
        self.ae = Model(inp, x_hat, name="DEC_AE")

    def pretrain(self, X: np.ndarray, epochs=100, batch_size=256, lr=1e-3):
        self.ae.compile(optimizer=optimizers.Adam(lr), loss="mse")
        self.ae.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)

    def _soft_assign(self, Z: np.ndarray) -> np.ndarray:
        """Student's t-kernel soft assignments."""
        dist = np.sum((Z[:, None] - self.centroids[None]) ** 2, axis=2)
        q = (1 + dist / self.alpha) ** (-(self.alpha + 1) / 2)
        q /= q.sum(axis=1, keepdims=True)
        return q

    def _target_distribution(self, Q: np.ndarray) -> np.ndarray:
        p = Q ** 2 / Q.sum(axis=0)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def fit(self, X: np.ndarray, max_iter=1000, batch_size=256,
            tol=0.001, update_interval=140):
        Z = self.encoder.predict(X, batch_size=batch_size, verbose=0)
        km = KMeans(n_clusters=self.K, n_init=20, random_state=42)
        km.fit(Z)
        self.centroids = km.cluster_centers_.astype(np.float32)

        # Fine-tune with clustering loss only
        opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
        prev_labels = None

        for it in range(max_iter):
            if it % update_interval == 0:
                Z = self.encoder.predict(X, batch_size=batch_size, verbose=0)
                Q = self._soft_assign(Z)
                P = self._target_distribution(Q)
                labels = Q.argmax(axis=1)

                if prev_labels is not None:
                    delta = (labels != prev_labels).mean()
                    if delta < tol:
                        break
                prev_labels = labels.copy()

            # Mini-batch update
            idx = np.random.choice(len(X), batch_size, replace=False)
            x_b = tf.constant(X[idx])
            p_b = tf.constant(P[idx])

            with tf.GradientTape() as tape:
                z_b = self.encoder(x_b, training=True)
                # KL loss approximation via soft assignments
                dist_b = tf.reduce_sum(
                    (tf.expand_dims(z_b, 1) -
                     tf.constant(self.centroids[None])) ** 2, axis=2)
                q_b = (1 + dist_b / self.alpha) ** (-(self.alpha + 1) / 2)
                q_b = q_b / (tf.reduce_sum(q_b, axis=1, keepdims=True) + 1e-9)
                kl_loss = tf.reduce_mean(
                    tf.reduce_sum(p_b * tf.math.log(p_b / (q_b + 1e-9) + 1e-9), axis=1)
                )

            grads = tape.gradient(kl_loss, self.encoder.trainable_variables)
            opt.apply_gradients(zip(grads, self.encoder.trainable_variables))

        return labels

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        Z = self.encoder.predict(X, batch_size=batch_size, verbose=0)
        Q = self._soft_assign(Z)
        return Q.argmax(axis=1)


# ---------------------------------------------------------------------------
# IDEC  (Guo et al. 2017 [7])
# ---------------------------------------------------------------------------

class IDEC(DEC):
    """
    Improved DEC — jointly optimises reconstruction + clustering loss
    with a fixed λ balance weight.
    """

    def fit(self, X: np.ndarray, max_iter=1000, batch_size=256,
            tol=0.001, update_interval=140, lam=0.1):

        Z = self.encoder.predict(X, batch_size=batch_size, verbose=0)
        km = KMeans(n_clusters=self.K, n_init=20, random_state=42)
        km.fit(Z)
        self.centroids = km.cluster_centers_.astype(np.float32)

        opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
        prev_labels = None

        for it in range(max_iter):
            if it % update_interval == 0:
                Z = self.encoder.predict(X, batch_size=batch_size, verbose=0)
                Q = self._soft_assign(Z)
                P = self._target_distribution(Q)
                labels = Q.argmax(axis=1)

                if prev_labels is not None and (labels != prev_labels).mean() < tol:
                    break
                prev_labels = labels.copy()

            idx = np.random.choice(len(X), batch_size, replace=False)
            x_b = tf.constant(X[idx])
            p_b = tf.constant(P[idx])

            ae_vars = (self.encoder.trainable_variables +
                       self.decoder.trainable_variables)

            with tf.GradientTape() as tape:
                z_b   = self.encoder(x_b, training=True)
                x_hat = self.decoder(z_b, training=True)

                recon_loss = tf.reduce_mean(tf.square(x_b - x_hat))

                dist_b = tf.reduce_sum(
                    (tf.expand_dims(z_b, 1) -
                     tf.constant(self.centroids[None])) ** 2, axis=2)
                q_b = (1 + dist_b / self.alpha) ** (-(self.alpha + 1) / 2)
                q_b = q_b / (tf.reduce_sum(q_b, axis=1, keepdims=True) + 1e-9)
                kl_loss = tf.reduce_mean(
                    tf.reduce_sum(p_b * tf.math.log(p_b / (q_b + 1e-9) + 1e-9), axis=1)
                )

                total = recon_loss + lam * kl_loss

            grads = tape.gradient(total, ae_vars)
            opt.apply_gradients(zip(grads, ae_vars))

        return labels


# ---------------------------------------------------------------------------
# DynAE  (Mrabah, Khan, Ksantini & Lachiri — Neural Networks 130:206-228, 2020
#         "Deep Clustering with a Dynamic Autoencoder: From Reconstruction
#         towards Centroids Construction")
# ---------------------------------------------------------------------------
#
# Reference implementation (image data, MNIST-style):
#   https://github.com/nairouz/DynAE/blob/master/DynAE/dynAE.ipynb
# That notebook trains a conv-AE on 28x28 images and isn't directly runnable
# on flattened household load-profile vectors — the class below reimplements
# the paper's core *mechanism* (not the exact conv architecture) on top of
# this file's existing build_flat_ae, so it fits the same (N, T) input the
# other baselines use.
#
# Core idea vs. DEC/IDEC: instead of a fixed reconstruction/clustering
# balance (IDEC's static `lam`), DynAE anneals a weight lambda(t) that
# starts at 1 (pure reconstruction) and decays smoothly towards 0
# (pure clustering/"centroids construction") over training. This is meant
# to avoid two failure modes: DEC's abrupt loss switch causing "Feature
# Randomness" (embedding collapse early on), and reconstruction's tendency
# to keep encoding non-discriminative detail ("Feature Drift") if never
# relaxed. See the paper for the full derivation — this is a pragmatic
# reimplementation of the described dynamics, not a line-for-line port.
# ---------------------------------------------------------------------------

class DynAE(IDEC):
    """
    Dynamic Autoencoder — same soft-assignment/KL clustering machinery as
    DEC/IDEC, but the reconstruction<->clustering balance lambda(t) is
    annealed smoothly across training instead of held fixed.

    lambda(t) = lambda_min + (1 - lambda_min) * 0.5 * (1 + cos(pi * t / T))
    i.e. a cosine decay from ~1 (t=0, reconstruction-dominated) down to
    lambda_min (t=T, clustering-dominated), so early iterations behave like
    a plain autoencoder and later iterations behave like DEC.
    """

    def fit(self, X: np.ndarray, max_iter=1000, batch_size=256,
            tol=0.001, update_interval=140, lambda_min=0.05):

        Z = self.encoder.predict(X, batch_size=batch_size, verbose=0)
        km = KMeans(n_clusters=self.K, n_init=20, random_state=42)
        km.fit(Z)
        self.centroids = km.cluster_centers_.astype(np.float32)

        opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
        prev_labels = None
        labels = None

        for it in range(max_iter):
            if it % update_interval == 0:
                Z = self.encoder.predict(X, batch_size=batch_size, verbose=0)
                Q = self._soft_assign(Z)
                P = self._target_distribution(Q)
                labels = Q.argmax(axis=1)

                if prev_labels is not None and (labels != prev_labels).mean() < tol:
                    break
                prev_labels = labels.copy()

            # Smooth cosine anneal: lambda≈1 early (reconstruction-heavy)
            # -> lambda≈lambda_min late (clustering-heavy).
            progress = min(it / max(max_iter - 1, 1), 1.0)
            lam = lambda_min + (1 - lambda_min) * 0.5 * (1 + np.cos(np.pi * progress))

            idx = np.random.choice(len(X), batch_size, replace=False)
            x_b = tf.constant(X[idx])
            p_b = tf.constant(P[idx])

            ae_vars = (self.encoder.trainable_variables +
                       self.decoder.trainable_variables)

            with tf.GradientTape() as tape:
                z_b   = self.encoder(x_b, training=True)
                x_hat = self.decoder(z_b, training=True)

                recon_loss = tf.reduce_mean(tf.square(x_b - x_hat))

                dist_b = tf.reduce_sum(
                    (tf.expand_dims(z_b, 1) -
                     tf.constant(self.centroids[None])) ** 2, axis=2)
                q_b = (1 + dist_b / self.alpha) ** (-(self.alpha + 1) / 2)
                q_b = q_b / (tf.reduce_sum(q_b, axis=1, keepdims=True) + 1e-9)
                kl_loss = tf.reduce_mean(
                    tf.reduce_sum(p_b * tf.math.log(p_b / (q_b + 1e-9) + 1e-9), axis=1)
                )

                # lambda(t)*reconstruction + (1-lambda(t))*clustering —
                # the "gradual elimination of reconstruction" from the paper.
                total = lam * recon_loss + (1 - lam) * kl_loss

            grads = tape.gradient(total, ae_vars)
            opt.apply_gradients(zip(grads, ae_vars))

        return labels


def run_dynae(X_flat:     np.ndarray,
              K:          int,
              latent_dim: int = 10,
              pretrain_epochs: int = 100,
              max_iter:   int = 1000,
              batch_size: int = 256,
              lr:         float = 1e-3) -> np.ndarray:
    """
    Convenience wrapper: build, pretrain, and fit DynAE end-to-end, matching
    the calling convention of run_lstm_ae / run_classical_baselines. Returns
    cluster labels (N,) — feed these into _compute_cluster_metrics(X_flat,
    labels) to get Silhouette/DBI/CHI on the same footing as the other
    baselines, or fold the result into run_deep_baselines() below.
    """
    model = DynAE(input_dim=X_flat.shape[1], K=K, latent_dim=latent_dim)
    model.pretrain(X_flat, epochs=pretrain_epochs, batch_size=batch_size, lr=lr)
    return model.fit(X_flat, max_iter=max_iter, batch_size=batch_size)


def run_deep_baselines(X_flat: np.ndarray,
                        K:      int,
                        latent_dim:      int = 10,
                        pretrain_epochs: int = 100,
                        max_iter:        int = 1000,
                        batch_size:      int = 256) -> dict:
    """
    Runs the deep baselines (DEC, IDEC, DynAE) — the counterpart to
    run_classical_baselines() for the K-Means/Hierarchical/GMM/Spectral/
    DBSCAN group. Same return shape: {name: {"labels", "sil", "dbi", "chi",
    "n_clusters"}}, so results merge straight into build_comparison_table().

    NOTE: unlike the classical baselines, these train neural networks —
    expect real wall-clock time (minutes, not seconds) even on a GPU runtime.
    """
    results = {}
    input_dim = X_flat.shape[1]

    builders = {
        "DEC":   lambda: DEC(input_dim=input_dim, K=K, latent_dim=latent_dim),
        "IDEC":  lambda: IDEC(input_dim=input_dim, K=K, latent_dim=latent_dim),
        "DynAE": lambda: DynAE(input_dim=input_dim, K=K, latent_dim=latent_dim),
    }

    for name, build in builders.items():
        print(f"[Deep] Running {name} …")
        try:
            model = build()
            model.pretrain(X_flat, epochs=pretrain_epochs, batch_size=batch_size)
            labels = model.fit(X_flat, max_iter=max_iter, batch_size=batch_size)

            metrics = _compute_cluster_metrics(X_flat, labels)
            results[name] = {"labels": labels, **metrics}
            sil, dbi, chi = metrics["sil"], metrics["dbi"], metrics["chi"]
            sil_s = "NaN" if np.isnan(sil) else f"{sil:.4f}"
            dbi_s = "NaN" if np.isnan(dbi) else f"{dbi:.4f}"
            chi_s = "NaN" if np.isnan(chi) else f"{chi:.1f}"
            print(f"  {name}: Silhouette={sil_s}  DBI={dbi_s}  CHI={chi_s}  "
                  f"(clusters: {metrics['n_clusters']})")

        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = {"labels": None, "sil": np.nan, "dbi": np.nan,
                              "chi": np.nan, "n_clusters": 0}

    return results# ===========================================================================
# ─── CLASSICAL BASELINES ─────────────────────────────────────────────────── #
# ===========================================================================

def _compute_cluster_metrics(X_flat: np.ndarray, labels: np.ndarray,
                              sil_sample_cap: int = 5000,
                              random_state: int = 42) -> dict:
    """
    Silhouette (capped-subsample, matches clustering.py's evaluate_metrics),
    Davies-Bouldin and Calinski-Harabasz (full data — these two are cheap
    and, unlike silhouette, not O(N^2), so no subsampling is needed).

    DBSCAN noise points (label == -1) are excluded from all three metrics,
    since they don't belong to any cluster.

    Returns {"sil": ..., "dbi": ..., "chi": ..., "n_clusters": ...}.
    Any metric that can't be computed (e.g. <2 populated clusters, or a
    cluster with a single member) is returned as NaN rather than raising.
    """
    mask = labels != -1
    X_eff, labels_eff = X_flat[mask], labels[mask]

    unique, counts = np.unique(labels_eff, return_counts=True)
    n_unique = len(unique)

    out = {"sil": np.nan, "dbi": np.nan, "chi": np.nan, "n_clusters": n_unique}
    if n_unique < 2 or counts.min() < 2 or len(labels_eff) <= n_unique:
        return out

    try:
        n = X_eff.shape[0]
        if n > sil_sample_cap:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, sil_sample_cap, replace=False)
            X_ss, L_ss = X_eff[idx], labels_eff[idx]
            if len(np.unique(L_ss)) < n_unique:      # subsample dropped a cluster
                X_ss, L_ss = X_eff, labels_eff
        else:
            X_ss, L_ss = X_eff, labels_eff
        out["sil"] = float(silhouette_score(X_ss, L_ss, metric="euclidean"))
    except Exception as e:
        print(f"    [Metrics] Silhouette error: {e}")

    try:
        out["dbi"] = float(davies_bouldin_score(X_eff, labels_eff))
    except Exception as e:
        print(f"    [Metrics] DBI error: {e}")

    try:
        out["chi"] = float(calinski_harabasz_score(X_eff, labels_eff))
    except Exception as e:
        print(f"    [Metrics] CHI error: {e}")

    return out


def run_classical_baselines(X_flat: np.ndarray,
                             K:      int,
                             eps_dbscan: float = 0.5,
                             min_samples: int  = 5) -> dict:
    """
    Run all five classical baselines on the flat feature matrix X_flat.

    Parameters
    ----------
    X_flat : (N, T)  — already normalised
    K      : number of clusters (for methods that require K)

    Returns
    -------
    results : {method_name: {"labels": ..., "sil": ..., "dbi": ..., "chi": ...}}
    """
    results = {}

    methods = {
        "K-Means":      KMeans(n_clusters=K, n_init=20, random_state=42),
        "Hierarchical": AgglomerativeClustering(n_clusters=K),
        "GMM":          GaussianMixture(n_components=K, random_state=42),
        "Spectral":     SpectralClustering(n_clusters=K, affinity="rbf",
                                           random_state=42, n_init=10),
        "DBSCAN":       DBSCAN(eps=eps_dbscan, min_samples=min_samples),
    }

    for name, model in methods.items():
        print(f"[Classical] Running {name} …")
        try:
            labels  = model.fit_predict(X_flat)
            metrics = _compute_cluster_metrics(X_flat, labels)

            results[name] = {"labels": labels, **metrics}
            sil, dbi, chi = metrics["sil"], metrics["dbi"], metrics["chi"]
            sil_s = "NaN" if np.isnan(sil) else f"{sil:.4f}"
            dbi_s = "NaN" if np.isnan(dbi) else f"{dbi:.4f}"
            chi_s = "NaN" if np.isnan(chi) else f"{chi:.1f}"
            print(f"  {name}: Silhouette={sil_s}  DBI={dbi_s}  CHI={chi_s}  "
                  f"(clusters: {metrics['n_clusters']})")

        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = {"labels": None, "sil": np.nan, "dbi": np.nan,
                              "chi": np.nan, "n_clusters": 0}

    return results


# ===========================================================================
# ─── ConvLSTM-DynAE (the paper's own model) ─────────────────────────────── #
# ===========================================================================

def load_dynae_metrics(results_dir, dataset: str, k: int = None) -> dict:
    """
    Pull the already-computed ConvLSTM-DynAE metrics out of
    `metrics_all_k_<dataset>.csv` (written by main.py) so the paper's own
    model can sit as a row in the same comparison table as the classical
    baselines — no retraining or re-clustering needed.

    Parameters
    ----------
    results_dir : results/<dataset>/  (the folder containing the CSV)
    dataset     : "london" or "irish" — used only for the filename
    k           : which K's row to pull. Default (None) = whichever row has
                  Optimal == True in the CSV (i.e. the model's selected K).

    Returns
    -------
    {"labels": None, "sil": ..., "dbi": ..., "chi": ..., "n_clusters": k}
    "labels" is always None here — the CSV only stores the metrics, not the
    per-household assignments (those live separately in cluster_labels.npy
    under cluster/K<k>/ if ever needed for label-agreement analyses).
    """
    csv_path = Path(results_dir) / f"metrics_all_k_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"[DynAE] {csv_path} not found — run the main pipeline "
            f"(main.py) for '{dataset}' at least once first."
        )

    df = pd.read_csv(csv_path)
    if k is not None:
        row = df[df["K"] == k]
        if row.empty:
            raise ValueError(f"[DynAE] K={k} not found in {csv_path}")
    else:
        row = df[df["Optimal"] == True]  # noqa: E712
        if row.empty:
            raise ValueError(f"[DynAE] No row flagged Optimal in {csv_path}")

    row = row.iloc[0]
    to_f = lambda v: float(v) if str(v) != "NaN" else np.nan
    return {
        "labels":     None,
        "sil":        to_f(row["Silhouette Score"]),
        "dbi":        to_f(row["Davies-Bouldin Index"]),
        "chi":        to_f(row["Calinski-Harabasz Index"]),
        "n_clusters": int(row["K"]),
    }


def build_comparison_table(classical_results: dict,
                            results_dir,
                            dataset: str,
                            dynae_k: int = None,
                            dynae_name: str = "ConvLSTM-DynAE (Ours)") -> "pd.DataFrame":
    """
    Merge the classical-baseline results with the paper's own
    ConvLSTM-DynAE model into a single ranked comparison table.

    Returns a DataFrame sorted best-to-worst by Silhouette (primary),
    ready to print or export (e.g. to LaTeX via .to_latex()).
    """
    dynae_metrics = load_dynae_metrics(results_dir, dataset, k=dynae_k)

    all_results = dict(classical_results)
    all_results[dynae_name] = dynae_metrics

    rows = []
    for name, r in all_results.items():
        rows.append({
            "Method":      name,
            "Clusters":    r.get("n_clusters", np.nan),
            "Silhouette":  r["sil"],
            "DBI":         r["dbi"],
            "CHI":         r["chi"],
        })

    table = pd.DataFrame(rows).set_index("Method")
    # Higher Silhouette is better -> sort descending; NaNs go last.
    table = table.sort_values("Silhouette", ascending=False, na_position="last")
    return table


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N, T = 100, 48 * 7
    X_flat = np.random.rand(N, T).astype(np.float32)

    print("=== Classical baselines ===")
    res = run_classical_baselines(X_flat, K=3)
    for m, r in res.items():
        print(f"  {m}: sil={r['sil']:.4f}  dbi={r['dbi']:.4f}  chi={r['chi']:.2f}")

    print("\n=== LSTM-AE ===")
    labels = run_lstm_ae(X_flat, K=3, epochs=2, batch_size=16)
    print(f"  Labels: {np.unique(labels, return_counts=True)}")

    print("\nbaselines.py smoke test passed.")
