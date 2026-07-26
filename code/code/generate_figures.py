"""
generate_figures.py
====================
Publication-quality REVISION figures for Scientific Reports — the
authoritative 14-figure spec (FIGURES-2), which SUPERSEDES an earlier,
inferred 6-figure version. Every figure below was explicitly specified
(exact content, exact filename); nothing here is guessed.

  Fig01_LatentSpace              - UMAP > t-SNE > PCA (fallback chain),
                                   centroids marked, confidence ellipses,
                                   Sil/K/dataset annotated
  Fig02_RepresentativeProfiles   - 5 households nearest each centroid,
                                   thin gray + thick centroid overlay
  Fig03_Reconstruction           - original vs reconstructed, 5 households
                                   across different clusters, MSE annotated
  Fig04_ClusterSizes             - bar chart, count + percentage per bar
  Fig05_ClusterTransitions       - transition heatmap between consecutive
                                   outer iterations (Sankey not available
                                   in a matplotlib-only environment -
                                   heatmap is the spec's own sanctioned
                                   fallback)
  Fig06_LatentDistribution       - violin plots, first 8 latent dims,
                                   split by cluster
  Fig07_ClusterCharacteristics   - radar chart, 8 engineered features,
                                   one polygon per cluster
  Fig08_Statistics               - auto-selected ANOVA/Kruskal-Wallis,
                                   p-values + effect sizes + Statistics_Table.csv
  Fig09_CentroidDistances        - pairwise centroid distance heatmap
  Fig10_Compactness              - within-cluster distance-to-centroid
                                   distributions (boxplot)
  Fig11_DistanceComparison       - intra- vs inter-cluster distance,
                                   grouped bar
  Fig12_SilhouetteDistribution   - full per-sample silhouette plot
  Fig13_DBIVisualization         - scatter + centroids + separation,
                                   explains why DBI is low
  Fig14_CHVisualization          - between- vs within-cluster variance

------------------------------------------------------------------------
Hard constraints honoured throughout:
  - Does NOT rerun training or clustering - reads only saved artifacts
    (encoder/decoder weights, cluster_labels.npy, centroids.npy, the
    cached preprocessed tensor X, and the raw `wide` DataFrame for
    time-aware feature engineering in Fig 7/8).
  - Does NOT change any reported metric - Sil/DBI/CHI shown here are
    recomputed read-only from the SAME saved labels/embeddings that
    already produced the numbers in metrics_all_k_<dataset>.csv, purely
    for display inside these figures.
  - Works for both London and Irish without any code change - every
    figure function takes `dataset`/`K` as parameters and organises
    output under Figures/Revision/<dataset>/.
  - Every figure saved as BOTH 300dpi PNG and vector PDF.
  - One independent function per figure; a single generate_revision_
    figures() orchestrator reuses already-computed Z/labels/centroids/
    features across all 14 rather than recomputing per figure.
------------------------------------------------------------------------

Usage
-----
    from generate_figures import generate_revision_figures
    generate_revision_figures(results_dir="results/london", cache_dir="data/cache")
    generate_revision_figures(results_dir="results/irish",  cache_dir="data/cache")
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
from typing import Optional, Tuple, List

from scipy import stats as scipy_stats
from sklearn.decomposition   import PCA
from sklearn.manifold        import TSNE
from sklearn.metrics         import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

try:
    import umap as umap_lib
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

from model              import build_autoencoder
from data_preprocessing import load_preprocessed


# ===========================================================================
# Shared style / palette - identical across every figure in the paper
# ===========================================================================
# ===========================================================================
# Shared style / palette - identical across every figure in the paper
# [MODIFIED — journal-quality requirement] Font family, exact font sizes
# (title 16-18, axes 14, ticks 12, legend 12), and top/right spines
# removed globally, per Scientific Reports formatting requirements.
# Colors, cluster IDs, computations, and plotting order are UNCHANGED.
# ===========================================================================
CLUSTER_COLORS = ["#2196F3", "#E53935", "#43A047", "#FB8C00",
                  "#8E24AA", "#00ACC1", "#F4511E", "#6D4C41", "#546E7A"]

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "DejaVu Sans"],
    "font.size":          12,
    "axes.titlesize":     17,
    "axes.titleweight":   "bold",
    "axes.labelsize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


def _save_figure(fig: plt.Figure, filename_stem: str, dataset: str,
                 results_dir: str) -> None:
    """Saves BOTH 300dpi PNG and vector PDF to
    Figures/Revision/<dataset>/<filename_stem>.{png,pdf}"""
    fig_dir = Path(results_dir) / "Figures" / "Revision" / dataset
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = fig_dir / f"{filename_stem}.png"
    pdf_path = fig_dir / f"{filename_stem}.pdf"
    fig.savefig(str(png_path), dpi=600, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    print(f"[Figures] Saved -> {png_path}")
    print(f"[Figures] Saved -> {pdf_path}")
    plt.close(fig)


def _cluster_color(k: int) -> str:
    return CLUSTER_COLORS[k % len(CLUSTER_COLORS)]


# ===========================================================================
# Auto-detection + artifact loading (no retraining, no re-clustering)
# ===========================================================================

def _detect_dataset_and_optimal_k(results_dir: str) -> Tuple[str, int]:
    matches = glob.glob(str(Path(results_dir) / "metrics_all_k_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No metrics_all_k_*.csv found in {results_dir} - run the "
            "normal pipeline (main.py) at least once first."
        )
    csv_path = matches[0]
    dataset = Path(csv_path).stem.replace("metrics_all_k_", "")
    df = pd.read_csv(csv_path, index_col="K")
    optimal_rows = df[df["Optimal"] == True]  # noqa: E712
    if optimal_rows.empty:
        raise ValueError(f"No row marked Optimal=True in {csv_path}")
    optimal_k = int(optimal_rows.index[0])
    print(f"[Figures] Auto-detected dataset={dataset}, optimal K={optimal_k}")
    return dataset, optimal_k


def load_run_artifacts(results_dir: str, cache_dir: str,
                       dataset: Optional[str] = None,
                       K: Optional[int] = None,
                       n_steps: int = 7, n_length: int = 48,
                       n_features: int = 1, latent_dim: int = 10,
                       batch_size: int = 256) -> dict:
    """Loads everything needed for all 14 figures from artifacts ALREADY
    saved by a normal main.py run. No retraining, no re-clustering."""
    if dataset is None or K is None:
        auto_dataset, auto_k = _detect_dataset_and_optimal_k(results_dir)
        dataset = dataset or auto_dataset
        K       = K or auto_k

    X, wide = load_preprocessed(cache_dir, dataset)
    if X is None:
        raise FileNotFoundError(
            f"No cached tensor for {dataset} in {cache_dir}.")

    k_dir = Path(results_dir) / "cluster" / f"K{K}"
    enc_path, dec_path = (k_dir / "encoder_clustered.weights.h5",
                          k_dir / "decoder_clustered.weights.h5")
    labels_path, cents_path = (k_dir / "cluster_labels.npy",
                               k_dir / "centroids.npy")
    label_hist_path = k_dir / "label_history.npy"

    for p in [enc_path, dec_path, labels_path, cents_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    encoder, decoder, _ = build_autoencoder(
        n_steps, n_length, n_features, latent_dim)
    _ = encoder(X[:2]); _ = decoder(encoder(X[:2]))
    encoder.load_weights(str(enc_path))
    decoder.load_weights(str(dec_path))

    labels    = np.load(str(labels_path))
    centroids = np.load(str(cents_path))
    label_history = (np.load(str(label_hist_path))
                     if label_hist_path.exists() else None)

    print(f"[Figures] Loaded {dataset} K={K}: X={X.shape}, "
          f"labels={labels.shape}, centroids={centroids.shape}, "
          f"label_history={'present' if label_history is not None else 'not saved'}")

    return {
        "dataset": dataset, "K": K, "X": X, "wide": wide,
        "encoder": encoder, "decoder": decoder,
        "labels": labels, "centroids": centroids,
        "label_history": label_history, "batch_size": batch_size,
    }


def _encode_full(encoder, X, batch_size=256) -> np.ndarray:
    parts = []
    for i in range(0, X.shape[0], batch_size):
        parts.append(encoder(X[i:i + batch_size], training=False).numpy())
    return np.nan_to_num(np.vstack(parts), nan=0.0)


# ===========================================================================
# Household-level feature engineering (Fig 7 / Fig 8) - computed from the
# RAW (pre-normalisation) `wide` DataFrame: index=timestamps, columns=
# household IDs. Standard power-systems definitions throughout.
# ===========================================================================

def compute_household_features(wide: pd.DataFrame,
                                household_ids: List) -> pd.DataFrame:
    """
    Returns one row per household with 8 standard load-profiling metrics:
      avg_daily_energy    : mean of daily-summed consumption (kWh/day)
      peak_demand          : single highest half-hourly reading (kW)
      night_consumption    : mean reading during 00:00-06:00
      evening_consumption  : mean reading during 17:00-21:00 (typical peak)
      weekend_ratio        : mean(Sat/Sun) / mean(Mon-Fri)
      winter_increase      : (mean(Winter) - mean(Summer)) / mean(Summer)
      peak_to_avg_ratio    : peak_demand / mean_reading (PAR - standard
                             power-systems metric)
      load_factor          : mean_reading / peak_demand (LF - standard
                             power-systems metric, inverse of PAR)
    """
    idx = wide.index
    hour = idx.hour
    dow  = idx.dayofweek         # 0=Mon ... 6=Sun
    month = idx.month

    night_mask   = (hour >= 0) & (hour < 6)
    evening_mask = (hour >= 17) & (hour < 21)
    weekend_mask = dow >= 5
    weekday_mask = ~weekend_mask
    winter_mask  = month.isin([12, 1, 2])
    summer_mask  = month.isin([6, 7, 8])

    rows = []
    for hh in household_ids:
        s = wide[hh]
        daily_sum = s.resample("D").sum()
        mean_reading = float(s.mean())
        peak = float(s.max())
        weekday_mean = float(s[weekday_mask].mean())
        weekend_mean = float(s[weekend_mask].mean())
        winter_mean = float(s[winter_mask].mean())
        summer_mean = float(s[summer_mask].mean())

        rows.append({
            "household": hh,
            "avg_daily_energy":    float(daily_sum.mean()),
            "peak_demand":         peak,
            "night_consumption":   float(s[night_mask].mean()),
            "evening_consumption": float(s[evening_mask].mean()),
            "weekend_ratio":       (weekend_mean / weekday_mean
                                    if weekday_mean > 0 else np.nan),
            "winter_increase":     ((winter_mean - summer_mean) / summer_mean
                                    if summer_mean > 0 else np.nan),
            "peak_to_avg_ratio":   (peak / mean_reading
                                    if mean_reading > 0 else np.nan),
            "load_factor":         (mean_reading / peak if peak > 0 else np.nan),
        })
    return pd.DataFrame(rows).set_index("household")


# ===========================================================================
# Fig 1 - Learned Latent Space (UMAP > t-SNE > PCA)
# ===========================================================================

def plot_latent_space(Z: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                      sil_score: float, dataset: str, K: int,
                      results_dir: str, sample_size: int = 3000,
                      random_state: int = 42) -> None:
    """Fig01_LatentSpace - one point per household, coloured by cluster,
    centroids highlighted, 2-sigma confidence ellipses, Sil/K/dataset
    annotated directly on the figure."""
    N = Z.shape[0]
    if N > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, sample_size, replace=False)
    else:
        idx = np.arange(N)
    Z_s, labels_s = Z[idx], labels[idx]

    method_used = None
    if _UMAP_AVAILABLE:
        try:
            reducer = umap_lib.UMAP(n_components=2, random_state=random_state)
            Z2_s = reducer.fit_transform(Z_s)
            Z2_cent = reducer.transform(centroids)
            method_used = "UMAP"
        except Exception as e:
            warnings.warn(f"UMAP failed ({e}), falling back to t-SNE.")
    if method_used is None:
        try:
            perplexity = min(30, max(5, len(Z_s) // 20))
            # Fit t-SNE jointly on samples + centroids so centroids land
            # in the SAME projected space (t-SNE has no separate .transform())
            combined = np.vstack([Z_s, centroids])
            tsne = TSNE(n_components=2, perplexity=perplexity,
                       random_state=random_state, init="pca")
            combined_2d = tsne.fit_transform(combined)
            Z2_s, Z2_cent = combined_2d[:len(Z_s)], combined_2d[len(Z_s):]
            method_used = "t-SNE"
        except Exception as e:
            warnings.warn(f"t-SNE failed ({e}), falling back to PCA.")
    if method_used is None:
        pca = PCA(n_components=2, random_state=random_state)
        Z2_s = pca.fit_transform(Z_s)
        Z2_cent = pca.transform(centroids)
        method_used = "PCA"

    fig, ax = plt.subplots(figsize=(8, 7))
    for k in range(K):
        mask = labels_s == k
        pts = Z2_s[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=_cluster_color(k), s=10,
                  alpha=0.55, edgecolors="none", label=f"Cluster {k}")

        # 2-sigma confidence ellipse
        if len(pts) >= 3:
            cov = np.cov(pts.T)
            mean_pt = pts.mean(axis=0)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2 * 2 * np.sqrt(np.maximum(eigvals, 0))  # 2-sigma
            ellipse = Ellipse(mean_pt, width, height, angle=angle,
                             facecolor="none", edgecolor=_cluster_color(k),
                             linewidth=1.5, linestyle="--", alpha=0.7)
            ax.add_patch(ellipse)

        # Centroid marker
        ax.scatter(Z2_cent[k, 0], Z2_cent[k, 1], c=_cluster_color(k),
                  s=260, marker="X", edgecolors="black", linewidths=1.5,
                  zorder=5)

    # [MODIFIED — reviewer requirement] Title no longer embeds the
    # Silhouette score or dataset/K context — uses the exact requested
    # title text. Kept method-accurate (UMAP/t-SNE/PCA) rather than
    # hardcoding "UMAP" unconditionally, since this figure falls back to
    # t-SNE or PCA when umap-learn isn't installed — labelling a PCA plot
    # "UMAP Visualization" would misrepresent which method actually ran.
    ax.set_title(f"{method_used} Visualization of Learned Latent Representations")
    ax.set_xlabel(f"{method_used} dimension 1")
    ax.set_ylabel(f"{method_used} dimension 2")
    ax.legend(fontsize=8, loc="best", markerscale=1.5)
    ax.grid(True, linestyle="--", alpha=0.25)
    _save_figure(fig, "Fig01_LatentSpace", dataset, results_dir)


# ===========================================================================
# Fig 2 - Representative Household Profiles
# ===========================================================================

def plot_representative_profiles(X: np.ndarray, Z: np.ndarray,
                                  labels: np.ndarray, centroids: np.ndarray,
                                  decoder, dataset: str, K: int,
                                  results_dir: str, n_reps: int = 5) -> None:
    """Fig02_RepresentativeProfiles - for each cluster, the 5 households
    CLOSEST to the centroid in latent space (not random), plotted as thin
    gray lines, with the decoded centroid overlaid as a thick coloured
    line. Demonstrates low intra-cluster variability directly."""
    fig, axes = plt.subplots(1, K, figsize=(4.2 * K, 4), squeeze=False)
    axes = axes[0]

    for k in range(K):
        idx_k = np.where(labels == k)[0]
        ax = axes[k]
        if len(idx_k) == 0:
            ax.axis("off")
            continue

        dists = np.linalg.norm(Z[idx_k] - centroids[k], axis=1)
        closest = idx_k[np.argsort(dists)[:n_reps]]

        for i, sample_idx in enumerate(closest):
            curve = X[sample_idx].reshape(-1)
            ax.plot(curve, color="gray", linewidth=0.9, alpha=0.6,
                   label="Representative households" if i == 0 else None)

        centroid_curve = decoder(centroids[k:k + 1], training=False).numpy().reshape(-1)
        ax.plot(centroid_curve, color=_cluster_color(k), linewidth=2.4,
               label="Cluster centroid")

        ax.set_title(f"Cluster {k}  (n={len(idx_k)})", fontsize=10)
        ax.set_xlabel("Half-hour index (7d x 48)", fontsize=8)
        if k == 0:
            ax.set_ylabel("Normalised kWh", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(f"Representative Household Profiles - "
               f"{dataset.title()} K={K}", fontsize=13, fontweight="bold", y=1.03)
    _save_figure(fig, "Fig02_RepresentativeProfiles", dataset, results_dir)


# ===========================================================================
# Fig 3 - Original vs Reconstructed Signals
# ===========================================================================

def plot_reconstruction_examples(X: np.ndarray, encoder, decoder,
                                 labels: np.ndarray, dataset: str, K: int,
                                 results_dir: str, n_households: int = 5,
                                 random_state: int = 42) -> None:
    """Fig03_Reconstruction - 5 households spanning different clusters,
    original + reconstructed overlaid on the SAME subplot, MSE annotated."""
    rng = np.random.default_rng(random_state)
    chosen = []
    clusters_present = np.unique(labels)
    # Spread the 5 choices across as many distinct clusters as possible
    for k in clusters_present:
        idx_k = np.where(labels == k)[0]
        if len(idx_k) > 0:
            chosen.append(int(rng.choice(idx_k)))
        if len(chosen) >= n_households:
            break
    # If K < n_households, fill remaining slots with random extra households
    while len(chosen) < n_households:
        extra = int(rng.integers(0, X.shape[0]))
        if extra not in chosen:
            chosen.append(extra)

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    axes = axes[0]

    for i, sample_idx in enumerate(chosen):
        x_orig = X[sample_idx:sample_idx + 1]
        x_hat  = decoder(encoder(x_orig, training=False), training=False).numpy()
        orig_flat, hat_flat = x_orig.reshape(-1), x_hat.reshape(-1)
        mse = float(np.mean((orig_flat - hat_flat) ** 2))
        k_label = int(labels[sample_idx])

        ax = axes[i]
        ax.plot(orig_flat, color=_cluster_color(k_label), linewidth=1.5,
               label="Original")
        ax.plot(hat_flat, color="black", linewidth=1.0, linestyle="--",
               label="Reconstructed")
        ax.set_title(f"Household #{sample_idx} (Cluster {k_label})", fontsize=9)
        ax.text(0.03, 0.95, f"MSE = {mse:.5f}", transform=ax.transAxes,
               fontsize=8, va="top",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        if i == 0:
            ax.set_ylabel("Normalised kWh", fontsize=9)
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xlabel("Half-hour index", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(f"Original vs Reconstructed Signals - "
               f"{dataset.title()} K={K}", fontsize=13, fontweight="bold", y=1.03)
    _save_figure(fig, "Fig03_Reconstruction", dataset, results_dir)


# ===========================================================================
# Fig 4 - Cluster Size Distribution
# ===========================================================================

def plot_cluster_sizes(labels: np.ndarray, dataset: str, K: int,
                       results_dir: str) -> None:
    """Fig04_ClusterSizes - bar chart, count + percentage annotated."""
    N = len(labels)
    counts = [int(np.sum(labels == k)) for k in range(K)]
    pcts   = [100.0 * c / N for c in counts]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(K), counts,
                 color=[_cluster_color(k) for k in range(K)],
                 edgecolor="black", linewidth=0.6)
    for k, (bar, c, p) in enumerate(zip(bars, counts, pcts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
               f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Cluster Size Distribution - {dataset.title()} K={K}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of households")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"Cluster {k}" for k in range(K)])
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")
    _save_figure(fig, "Fig04_ClusterSizes", dataset, results_dir)


# ===========================================================================
# Fig 5 - Cluster Assignment Evolution (transition heatmap)
# ===========================================================================

def plot_cluster_transitions(label_history: np.ndarray, dataset: str,
                             K: int, results_dir: str) -> None:
    """Fig05_ClusterTransitions - transition heatmap between consecutive
    outer iterations (Sankey requires plotly; heatmap is the spec's own
    sanctioned fallback for a matplotlib-only environment). One heatmap
    panel per consecutive iteration pair, annotated with household counts."""
    n_outer = label_history.shape[0]
    if n_outer < 2:
        print("[Figures] Skipping Fig05 - fewer than 2 outer iterations "
              "recorded, nothing to show a transition between.")
        return

    n_transitions = n_outer - 1
    fig, axes = plt.subplots(1, n_transitions,
                             figsize=(4.5 * n_transitions, 4), squeeze=False)
    axes = axes[0]

    for t in range(n_transitions):
        labels_t, labels_t1 = label_history[t], label_history[t + 1]
        transition_matrix = np.zeros((K, K), dtype=int)
        for i in range(len(labels_t)):
            transition_matrix[labels_t[i], labels_t1[i]] += 1

        ax = axes[t]
        im = ax.imshow(transition_matrix, cmap="Blues", aspect="auto")
        for i in range(K):
            for j in range(K):
                val = transition_matrix[i, j]
                color = "white" if val > transition_matrix.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                       fontsize=8, color=color)
        n_changed = int(np.sum(labels_t != labels_t1))
        ax.set_title(f"Outer {t+1} -> {t+2}\n"
                   f"{n_changed} households changed cluster", fontsize=9)
        ax.set_xlabel(f"Cluster at iter {t+2}", fontsize=8)
        if t == 0:
            ax.set_ylabel(f"Cluster at iter {t+1}", fontsize=8)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Cluster Assignment Evolution - "
               f"{dataset.title()} K={K}", fontsize=13, fontweight="bold", y=1.05)
    _save_figure(fig, "Fig05_ClusterTransitions", dataset, results_dir)


# ===========================================================================
# Fig 6 - Latent Feature Distribution
# ===========================================================================

def plot_latent_feature_distribution(Z: np.ndarray, labels: np.ndarray,
                                     dataset: str, K: int,
                                     results_dir: str,
                                     n_dims: int = 8) -> None:
    """Fig06_LatentDistribution - violin plots for the first n_dims latent
    dimensions, split by cluster, showing latent representation
    separability dimension-by-dimension."""
    n_dims = min(n_dims, Z.shape[1])
    n_cols = (n_dims + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), squeeze=False)
    axes = axes.flatten()

    for d in range(n_dims):
        ax = axes[d]
        data_per_cluster = [Z[labels == k, d] for k in range(K)]
        parts = ax.violinplot(data_per_cluster, positions=range(K),
                             showmeans=True, showextrema=True)
        for k, body in enumerate(parts["bodies"]):
            body.set_facecolor(_cluster_color(k))
            body.set_alpha(0.65)
        ax.set_title(f"Latent dim {d}", fontsize=10)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"C{k}" for k in range(K)], fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.25, axis="y")

    for d in range(n_dims, len(axes)):
        axes[d].axis("off")

    fig.suptitle(f"Latent Feature Distribution (first {n_dims} dims) - "
               f"{dataset.title()} K={K}", fontsize=13, fontweight="bold", y=1.02)
    _save_figure(fig, "Fig06_LatentDistribution", dataset, results_dir)


# ===========================================================================
# Fig 7 - Cluster Characteristics Summary (radar chart)
# ===========================================================================

def plot_cluster_characteristics_radar(features_df: pd.DataFrame,
                                       labels: np.ndarray, dataset: str,
                                       K: int, results_dir: str) -> None:
    """Fig07_ClusterCharacteristics - radar chart, one polygon per cluster,
    over the 8 engineered features (min-max normalised across clusters
    so all axes share a comparable [0,1] scale)."""
    feature_names = list(features_df.columns)
    n_feat = len(feature_names)

    cluster_means = pd.DataFrame({
        k: features_df.loc[labels == k, feature_names].mean()
        for k in range(K)
    }).T   # rows=clusters, cols=features

    # Min-max normalise each feature ACROSS clusters (not across households)
    normed = (cluster_means - cluster_means.min()) / \
             (cluster_means.max() - cluster_means.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for k in range(K):
        values = normed.loc[k].tolist()
        values += values[:1]
        ax.plot(angles, values, color=_cluster_color(k), linewidth=2,
               label=f"Cluster {k}")
        ax.fill(angles, values, color=_cluster_color(k), alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title(f"Cluster Characteristics Summary - "
               f"{dataset.title()} K={K}\n(normalised across clusters)",
               fontsize=12, fontweight="bold", y=1.08)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save_figure(fig, "Fig07_ClusterCharacteristics", dataset, results_dir)


# ===========================================================================
# Fig 8 - Statistical Validation
# ===========================================================================

def statistical_validation(features_df: pd.DataFrame, labels: np.ndarray,
                           dataset: str, K: int, results_dir: str) -> None:
    """Fig08_Statistics + Statistics_Table.csv - for each of 5 specified
    features, auto-selects ANOVA (if all groups pass a Shapiro-Wilk
    normality check) or Kruskal-Wallis (otherwise), reports p-value and
    an effect size (eta-squared for ANOVA, epsilon-squared for KW)."""
    test_features = ["avg_daily_energy", "peak_demand", "load_factor",
                     "weekend_ratio", "winter_increase"]
    test_features = [f for f in test_features if f in features_df.columns]

    results = []
    for feat in test_features:
        groups = [features_df.loc[labels == k, feat].dropna().values
                 for k in range(K)]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue

        # Normality check per group (Shapiro-Wilk, capped sample for speed)
        normal = True
        for g in groups:
            sample = g if len(g) <= 500 else np.random.choice(g, 500, replace=False)
            try:
                _, p_norm = scipy_stats.shapiro(sample)
                if p_norm < 0.05:
                    normal = False
                    break
            except Exception:
                normal = False
                break

        n_total = sum(len(g) for g in groups)
        if normal:
            stat, p_val = scipy_stats.f_oneway(*groups)
            grand_mean = np.concatenate(groups).mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total   = sum(((g - grand_mean) ** 2).sum() for g in groups)
            effect_size = ss_between / ss_total if ss_total > 0 else np.nan
            test_name = "ANOVA"
            effect_name = "eta-squared"
        else:
            stat, p_val = scipy_stats.kruskal(*groups)
            effect_size = ((stat - len(groups) + 1) / (n_total - len(groups))
                           if n_total > len(groups) else np.nan)
            test_name = "Kruskal-Wallis"
            effect_name = "epsilon-squared"

        sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
              else "*" if p_val < 0.05 else "ns")

        results.append({
            "feature": feat, "test": test_name, "statistic": stat,
            "p_value": p_val, "significance": sig,
            "effect_size_type": effect_name, "effect_size": effect_size,
        })

    stats_df = pd.DataFrame(results)
    csv_path = Path(results_dir) / "Figures" / "Revision" / dataset / "Statistics_Table.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(str(csv_path), index=False)
    print(f"[Figures] Saved -> {csv_path}")

    if stats_df.empty:
        print("[Figures] Skipping Fig08 plot - no valid feature comparisons.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    neg_log_p = -np.log10(stats_df["p_value"].clip(lower=1e-300))
    bars = ax.bar(stats_df["feature"], neg_log_p,
                 color=["#43A047" if s != "ns" else "#B0BEC5"
                       for s in stats_df["significance"]],
                 edgecolor="black", linewidth=0.6)
    for bar, sig, eff in zip(bars, stats_df["significance"], stats_df["effect_size"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
               f"{sig}\neffect={eff:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(y=-np.log10(0.05), color="red", linestyle="--", linewidth=1,
             label="p=0.05 threshold")
    ax.set_title(f"Statistical Validation Between Clusters - "
               f"{dataset.title()} K={K}\n"
               f"(test auto-selected per feature: ANOVA if normal, else Kruskal-Wallis)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_xlabel("Feature")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")
    _save_figure(fig, "Fig08_Statistics", dataset, results_dir)


# ===========================================================================
# Fig 9 - Centroid Distance Matrix
# ===========================================================================

def plot_centroid_distance_matrix(centroids: np.ndarray, dataset: str,
                                  K: int, results_dir: str) -> None:
    """Fig09_CentroidDistances - pairwise Euclidean distances between
    latent centroids, annotated heatmap with colorbar."""
    dist_matrix = euclidean_distances(centroids)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(dist_matrix, cmap="viridis")
    for i in range(K):
        for j in range(K):
            val = dist_matrix[i, j]
            color = "white" if val < dist_matrix.max() / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   fontsize=9, color=color)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"C{k}" for k in range(K)])
    ax.set_yticklabels([f"C{k}" for k in range(K)])
    ax.set_title(f"Centroid Distance Matrix - {dataset.title()} K={K}")
    fig.colorbar(im, ax=ax, label="Euclidean distance (latent space)")
    _save_figure(fig, "Fig09_CentroidDistances", dataset, results_dir)


# ===========================================================================
# Fig 10 - Cluster Compactness
# ===========================================================================

def plot_cluster_compactness(Z: np.ndarray, labels: np.ndarray,
                             centroids: np.ndarray, dataset: str, K: int,
                             results_dir: str) -> None:
    """Fig10_Compactness - within-cluster distance-to-own-centroid
    distributions, boxplot per cluster."""
    distances_per_cluster = []
    for k in range(K):
        idx_k = np.where(labels == k)[0]
        d = np.linalg.norm(Z[idx_k] - centroids[k], axis=1)
        distances_per_cluster.append(d)

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(distances_per_cluster, positions=range(K),
                   patch_artist=True, showfliers=True)
    for k, box in enumerate(bp["boxes"]):
        box.set_facecolor(_cluster_color(k))
        box.set_alpha(0.65)

    ax.set_title(f"Cluster Compactness - {dataset.title()} K={K}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Distance to own centroid (latent space)")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"Cluster {k}" for k in range(K)])
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")
    _save_figure(fig, "Fig10_Compactness", dataset, results_dir)


# ===========================================================================
# Fig 11 - Inter- vs Intra-Cluster Distance Comparison
# ===========================================================================

def plot_inter_intra_distance_comparison(Z: np.ndarray, labels: np.ndarray,
                                         centroids: np.ndarray, dataset: str,
                                         K: int, results_dir: str) -> None:
    """Fig11_DistanceComparison - average intra-cluster distance vs
    average distance to the NEAREST other centroid, grouped bar per cluster."""
    centroid_dist_matrix = euclidean_distances(centroids)
    np.fill_diagonal(centroid_dist_matrix, np.inf)

    intra_avg, inter_avg = [], []
    for k in range(K):
        idx_k = np.where(labels == k)[0]
        intra_avg.append(float(np.mean(np.linalg.norm(
            Z[idx_k] - centroids[k], axis=1))) if len(idx_k) > 0 else np.nan)
        inter_avg.append(float(centroid_dist_matrix[k].min()))

    x = np.arange(K)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, intra_avg, width, label="Avg intra-cluster distance",
          color="#90CAF9", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, inter_avg, width,
          label="Avg distance to nearest centroid",
          color="#EF9A9A", edgecolor="black", linewidth=0.6)

    ax.set_title(f"Inter- vs Intra-Cluster Distance - "
               f"{dataset.title()} K={K}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Distance (latent space)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {k}" for k in range(K)])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")
    _save_figure(fig, "Fig11_DistanceComparison", dataset, results_dir)


# ===========================================================================
# Fig 12 - Silhouette Distribution (standard clustering-paper figure)
# ===========================================================================

def plot_silhouette_distribution(Z: np.ndarray, labels: np.ndarray,
                                 dataset: str, K: int, results_dir: str,
                                 sample_size: int = 5000,
                                 random_state: int = 42) -> float:
    """Fig12_SilhouetteDistribution - full per-sample silhouette plot,
    one colour per cluster, sorted within each group, dashed vertical
    line at the overall mean. Returns the mean Silhouette for reuse by
    Fig01's annotation."""
    N = Z.shape[0]
    if N > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, sample_size, replace=False)
        Z_s, labels_s = Z[idx], labels[idx]
    else:
        Z_s, labels_s = Z, labels

    sample_sil = silhouette_samples(Z_s, labels_s, metric="euclidean")
    overall_sil = float(np.mean(sample_sil))

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for k in range(K):
        k_sil = np.sort(sample_sil[labels_s == k])
        size_k = len(k_sil)
        y_upper = y_lower + size_k
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, k_sil,
                        facecolor=_cluster_color(k), edgecolor=_cluster_color(k),
                        alpha=0.75)
        ax.text(-0.05, y_lower + 0.5 * size_k, f"Cluster {k}", fontsize=9,
               va="center", ha="right")
        y_lower = y_upper + 10

    ax.axvline(x=overall_sil, color="black", linestyle="--", linewidth=1.3,
             label=f"Mean Silhouette = {overall_sil:.4f}")
    ax.set_title(f"Silhouette Distribution - {dataset.title()} K={K}\n"
               f"(n={len(Z_s)} households)")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Households (grouped by cluster)")
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.25, axis="x")
    _save_figure(fig, "Fig12_SilhouetteDistribution", dataset, results_dir)
    return overall_sil


# ===========================================================================
# Fig 13 - Davies-Bouldin Components
# ===========================================================================

def plot_dbi_visualization(Z: np.ndarray, labels: np.ndarray,
                           centroids: np.ndarray, dbi_score: float,
                           dataset: str, K: int, results_dir: str,
                           sample_size: int = 3000,
                           random_state: int = 42) -> None:
    """Fig13_DBIVisualization - PCA scatter with centroids and
    nearest-neighbour separation lines, explaining why DBI is low
    (tight within-cluster scatter relative to large between-centroid
    separation is exactly the ratio DBI measures)."""
    N = Z.shape[0]
    if N > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, sample_size, replace=False)
    else:
        idx = np.arange(N)
    Z_s, labels_s = Z[idx], labels[idx]

    pca = PCA(n_components=2, random_state=random_state)
    Z2_s = pca.fit_transform(Z_s)
    Z2_cent = pca.transform(centroids)

    centroid_dist_matrix = euclidean_distances(centroids)
    np.fill_diagonal(centroid_dist_matrix, np.inf)

    fig, ax = plt.subplots(figsize=(8, 7))
    for k in range(K):
        mask = labels_s == k
        ax.scatter(Z2_s[mask, 0], Z2_s[mask, 1], c=_cluster_color(k), s=8,
                  alpha=0.4, edgecolors="none")
        ax.scatter(Z2_cent[k, 0], Z2_cent[k, 1], c=_cluster_color(k), s=280,
                  marker="X", edgecolors="black", linewidths=1.5, zorder=5)

    # Draw a line to each centroid's nearest neighbour, annotating the gap
    for k in range(K):
        nearest = int(np.argmin(centroid_dist_matrix[k]))
        ax.plot([Z2_cent[k, 0], Z2_cent[nearest, 0]],
               [Z2_cent[k, 1], Z2_cent[nearest, 1]],
               color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_title(f"Davies-Bouldin Index Components - "
               f"{dataset.title()} K={K}\nDBI={dbi_score:.4f} "
               f"(lower = tighter clusters relative to their separation)")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.grid(True, linestyle="--", alpha=0.25)
    _save_figure(fig, "Fig13_DBIVisualization", dataset, results_dir)


# ===========================================================================
# Fig 14 - Calinski-Harabasz Interpretation
# ===========================================================================

def plot_ch_visualization(Z: np.ndarray, labels: np.ndarray,
                          centroids: np.ndarray, chi_score: float,
                          dataset: str, K: int, results_dir: str) -> None:
    """Fig14_CHVisualization - between-cluster variance vs within-cluster
    variance, the two quantities whose ratio (scaled by degrees of
    freedom) defines the Calinski-Harabasz index."""
    grand_mean = Z.mean(axis=0)
    between_var, within_var = [], []
    for k in range(K):
        idx_k = np.where(labels == k)[0]
        n_k = len(idx_k)
        between_var.append(n_k * np.sum((centroids[k] - grand_mean) ** 2))
        within_var.append(np.sum((Z[idx_k] - centroids[k]) ** 2))

    x = np.arange(K)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, between_var, width,
          label="Between-cluster variance (weighted)",
          color="#66BB6A", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, within_var, width,
          label="Within-cluster variance",
          color="#EF5350", edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_title(f"Calinski-Harabasz Components - "
               f"{dataset.title()} K={K}\nCHI={chi_score:.1f} "
               f"(higher = between-variance dominates within-variance)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Variance (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {k}" for k in range(K)])
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25, axis="y")
    _save_figure(fig, "Fig14_CHVisualization", dataset, results_dir)


# ===========================================================================
# Orchestrator - generates all 14 figures, reusing computed variables
# ===========================================================================

def generate_revision_figures(results_dir: str, cache_dir: str,
                              dataset: Optional[str] = None,
                              K: Optional[int] = None,
                              n_steps: int = 7, n_length: int = 48,
                              n_features: int = 1, latent_dim: int = 10,
                              batch_size: int = 256) -> None:
    """
    Generates all 14 revision figures (Fig01-Fig14) in one call, reusing
    already-computed embeddings/labels/centroids/features across figures
    rather than recomputing per-figure. Auto-detects dataset and optimal
    K if not given. Does NOT retrain or re-cluster - reads only
    already-saved artifacts.
    """
    art = load_run_artifacts(
        results_dir, cache_dir, dataset=dataset, K=K,
        n_steps=n_steps, n_length=n_length, n_features=n_features,
        latent_dim=latent_dim, batch_size=batch_size,
    )
    dataset, K = art["dataset"], art["K"]
    X, wide, encoder, decoder = art["X"], art["wide"], art["encoder"], art["decoder"]
    labels, centroids = art["labels"], art["centroids"]

    print(f"\n{'='*60}\nGenerating 14 revision figures - "
          f"{dataset.upper()} K={K}\n{'='*60}")

    # Shared, computed ONCE and reused across every figure that needs it
    Z = _encode_full(encoder, X, batch_size)
    sil_score = float(silhouette_score(
        Z, labels, sample_size=min(5000, len(Z)), random_state=42))
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
    dbi_score = float(davies_bouldin_score(Z, labels))
    chi_score = float(calinski_harabasz_score(Z, labels))

    household_ids = list(wide.columns[:X.shape[0]])
    features_df = compute_household_features(wide, household_ids)

    plot_latent_space(Z, labels, centroids, sil_score, dataset, K, results_dir)
    plot_representative_profiles(X, Z, labels, centroids, decoder, dataset, K, results_dir)
    plot_reconstruction_examples(X, encoder, decoder, labels, dataset, K, results_dir)
    plot_cluster_sizes(labels, dataset, K, results_dir)

    if art["label_history"] is not None:
        plot_cluster_transitions(art["label_history"], dataset, K, results_dir)
    else:
        print("[Figures] Skipping Fig05 - label_history.npy not found "
              "(this run predates that addition; re-run clustering to enable it).")

    plot_latent_feature_distribution(Z, labels, dataset, K, results_dir)
    plot_cluster_characteristics_radar(features_df, labels, dataset, K, results_dir)
    statistical_validation(features_df, labels, dataset, K, results_dir)
    plot_centroid_distance_matrix(centroids, dataset, K, results_dir)
    plot_cluster_compactness(Z, labels, centroids, dataset, K, results_dir)
    plot_inter_intra_distance_comparison(Z, labels, centroids, dataset, K, results_dir)
    plot_silhouette_distribution(Z, labels, dataset, K, results_dir)
    plot_dbi_visualization(Z, labels, centroids, dbi_score, dataset, K, results_dir)
    plot_ch_visualization(Z, labels, centroids, chi_score, dataset, K, results_dir)

    print(f"\n[Figures] All revision figures saved to "
          f"{results_dir}/Figures/Revision/{dataset}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate the 14 revision figures")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--cache_dir",   required=True)
    p.add_argument("--dataset",     default=None)
    p.add_argument("--K",           type=int, default=None)
    args = p.parse_args()
    generate_revision_figures(results_dir=args.results_dir,
                              cache_dir=args.cache_dir,
                              dataset=args.dataset, K=args.K)
