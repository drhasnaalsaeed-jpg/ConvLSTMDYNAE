"""
baselines.py
============
Baseline models referenced in the manuscript.

Deep baselines:
  - DEC
  - IDEC
  - DynAE
  - LSTM-AE

Classical baselines:
  - K-Means, Agglomerative, DBSCAN, Spectral, GMM

Evaluation protocol
-------------------
Deep clustering methods are evaluated in their respective learned latent
representations using the same internal validity indices:
Silhouette, Davies-Bouldin (DBI), and Calinski-Harabasz (CHI).

Classical methods are evaluated in the normalized input space because they
do not learn an explicit latent representation.

Important: "respective learned latent spaces" does not mean that all deep
models share an identical embedding geometry.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ===========================================================================
# EVALUATION HELPERS
# ===========================================================================

def _compute_latent_metrics(
    Z: np.ndarray,
    labels: np.ndarray,
    sil_sample_cap: int = 5000,
    random_state: int = 42,
) -> dict:
    """
    Compute internal clustering validity indices in the learned latent
    representation of a deep model.

    Silhouette:
        - Euclidean distance.
        - Reproducible subsampling is used only when N > sil_sample_cap.

    DBI and CHI:
        - Computed on the full effective latent representation.

    Noise points (label == -1), if present, are excluded.

    Returns
    -------
    dict with keys:
        sil, dbi, chi, n_clusters
    """
    Z = np.asarray(Z)
    labels = np.asarray(labels)

    mask = labels != -1
    Z_eff = Z[mask]
    labels_eff = labels[mask]

    unique, counts = np.unique(labels_eff, return_counts=True)
    n_unique = len(unique)

    out = {
        "sil": np.nan,
        "dbi": np.nan,
        "chi": np.nan,
        "n_clusters": n_unique,
    }

    if n_unique < 2 or len(labels_eff) <= n_unique or counts.min() < 2:
        return out
