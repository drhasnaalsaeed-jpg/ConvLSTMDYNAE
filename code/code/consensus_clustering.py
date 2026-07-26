"""
consensus_clustering.py
========================
Combines multiple INDEPENDENT ConvLSTM-DynAE runs (different random seeds)
into a single, more stable consensus clustering — rather than picking one
"best-looking" seed and reporting only that.

Why this is legitimate (and different from seed-shopping)
-----------------------------------------------------------
Reporting the single highest-Silhouette seed out of N tried is a form of
selection bias unless explicitly disclosed as "best of N seeds". Consensus
clustering instead COMBINES all N seeds' opinions into one partition:

  1. For each seed, run the full pretrain+cluster pipeline at a fixed K.
  2. Build a co-association matrix C (N_households x N_households), where
     C[i,j] = fraction of seeds that placed household i and j in the same
     cluster.
  3. Treat (1 - C) as a distance matrix and run agglomerative clustering
     on it to get the final consensus labels.

This is standard practice (Strehl & Ghosh 2002 "Cluster Ensembles";
Fred & Jain 2005 "Combining Multiple Clusterings using Evidence
Accumulation") and tends to produce MORE stable, better-separated final
partitions than any single noisy run, because households that are
genuinely similar get grouped together consistently across seeds, while
ambiguous/boundary households get "voted" into whichever cluster the
majority of seeds agree on.

Usage
-----
  python consensus_clustering.py \
      --dataset       london \
      --data_path     data/london/LCL-FullData.csv \
      --cache_dir     data/cache \
      --seed_dirs     results/london_sweep_42 results/london_sweep_21 results/london_sweep_7 \
      --K             3 \
      --output_dir    results/london_consensus_K3

Or from Python:
  from consensus_clustering import run_consensus_clustering
  run_consensus_clustering(dataset="london", data_path=..., cache_dir=...,
                           seed_dirs=[...], K=3, output_dir=...)
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

from model              import build_autoencoder
from data_preprocessing import load_preprocessed
from clustering          import encode_full


# ===========================================================================
# Co-association matrix
# ===========================================================================

def build_coassociation_matrix(labels_list: List[np.ndarray]) -> np.ndarray:
    """
    C[i,j] = fraction of runs in which household i and household j were
    assigned to the SAME cluster. Diagonal is always 1.0 (self-agreement).

    labels_list : list of (N,) label arrays, one per seed. All must have
                  the same N (same households, same order).
    Returns (N, N) float32 co-association matrix.
    """
    n_runs = len(labels_list)
    N      = len(labels_list[0])
    C      = np.zeros((N, N), dtype=np.float32)

    for labels in labels_list:
        same = (labels[:, None] == labels[None, :])
        C   += same.astype(np.float32)

    C /= n_runs
    return C


def consensus_labels_from_coassociation(C: np.ndarray, K: int,
                                        linkage: str = "average") -> np.ndarray:
    """
    Final consensus partition: agglomerative clustering on the
    co-association-derived distance matrix (1 - C).

    linkage='average' is standard for consensus clustering (robust to the
    fact that (1-C) is a similarity-derived, not a true metric, distance).
    """
    distance = 1.0 - C
    np.fill_diagonal(distance, 0.0)   # numerical safety — should already be ~0

    model = AgglomerativeClustering(
        n_clusters = K,
        metric     = "precomputed",
        linkage    = linkage,
    )
    labels = model.fit_predict(distance)
    return labels


def cluster_stability_scores(C: np.ndarray, labels: np.ndarray, K: int) -> dict:
    """
    Mean within-cluster co-association per consensus cluster — a direct
    measure of how CONSISTENTLY households in that cluster were grouped
    together across all seeds. 1.0 = perfect agreement across every seed;
    values near 1/K suggest that cluster's membership is essentially
    random noise (no real agreement between seeds).
    """
    scores = {}
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 2:
            scores[k] = float("nan")
            continue
        sub = C[np.ix_(idx, idx)]
        # Exclude diagonal (always 1.0, would bias the mean upward)
        off_diag = sub[~np.eye(len(idx), dtype=bool)]
        scores[k] = float(off_diag.mean()) if len(off_diag) else float("nan")
    return scores


# ===========================================================================
# Metric evaluation for consensus labels
# ===========================================================================

def evaluate_consensus_metrics(Z_reference: np.ndarray,
                               consensus_labels: np.ndarray) -> dict:
    """
    Consensus labels have no single native embedding (they come from a
    distance matrix, not a latent space). Metrics are computed in the
    REFERENCE embedding space — by default, the best individual seed's
    encoder output — which is standard practice when evaluating a
    consensus/ensemble partition against a representative feature space.
    """
    unique, counts = np.unique(consensus_labels, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan}

    N = Z_reference.shape[0]
    cap = min(5000, N)
    if N > cap:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(N, cap, replace=False)
        Zs, Ls = Z_reference[idx], consensus_labels[idx]
        if len(np.unique(Ls)) < len(unique):
            Zs, Ls = Z_reference, consensus_labels
    else:
        Zs, Ls = Z_reference, consensus_labels

    sil = float(silhouette_score(Zs, Ls, metric="euclidean"))
    dbi = float(davies_bouldin_score(Z_reference, consensus_labels))
    chi = float(calinski_harabasz_score(Z_reference, consensus_labels))
    return {"silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi}


# ===========================================================================
# Main entry point
# ===========================================================================

def run_consensus_clustering(dataset:    str,
                             cache_dir:  str,
                             seed_dirs:  List[str],
                             K:          int,
                             output_dir: str,
                             n_steps:    int = 7,
                             n_length:   int = 48,
                             n_features: int = 1,
                             latent_dim: int = 10,
                             batch_size: int = 256) -> dict:
    """
    Build a consensus clustering from N independently-trained seed runs.

    Parameters
    ----------
    seed_dirs : list of result directories, one per seed, each expected to
               contain cluster/K<K>/cluster_labels.npy and
               cluster/K<K>/encoder_clustered.weights.h5 (i.e. the output
               of a normal main.py run with --output_dir set to that path).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Consensus] Loading cached data for {dataset} ...")
    X, wide = load_preprocessed(cache_dir, dataset)
    if X is None:
        raise FileNotFoundError(
            f"No cached tensor found for {dataset} in {cache_dir}. "
            "Run the normal pipeline (main.py) at least once first."
        )
    print(f"[Consensus] X shape: {X.shape}")

    # ── Load each seed's labels ─────────────────────────────────────────
    labels_list  = []
    seed_sils    = []   # each seed's OWN individual silhouette, for reference
    seed_encoders = []

    for sd in seed_dirs:
        sd = Path(sd)
        labels_path = sd / "cluster" / f"K{K}" / "cluster_labels.npy"
        enc_path    = sd / "cluster" / f"K{K}" / "encoder_clustered.weights.h5"

        if not labels_path.exists():
            print(f"[Consensus] WARNING: missing {labels_path} — skipping {sd}")
            continue

        labels = np.load(str(labels_path))
        if len(labels) != X.shape[0]:
            print(f"[Consensus] WARNING: {sd} has {len(labels)} labels but "
                  f"X has {X.shape[0]} samples — skipping (mismatched run).")
            continue

        labels_list.append(labels)

        # Re-encode with this seed's clustered encoder to get its own Z
        # (needed to pick the "best" seed as the metric reference space,
        # and to report each seed's individual silhouette for comparison)
        enc, _, _ = build_autoencoder(n_steps, n_length, n_features, latent_dim)
        _ = enc(X[:2])
        enc.load_weights(str(enc_path))
        Z_seed = encode_full(enc, X, batch_size)

        unique = np.unique(labels)
        if len(unique) >= 2:
            cap = min(5000, len(Z_seed))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Z_seed), cap, replace=False) if len(Z_seed) > cap else np.arange(len(Z_seed))
            sil = float(silhouette_score(Z_seed[idx], labels[idx]))
        else:
            sil = float("nan")

        seed_sils.append(sil)
        seed_encoders.append((sd, enc, Z_seed))
        print(f"[Consensus] Loaded {sd.name}: K={K}, "
              f"individual Sil={sil:.4f}")

    if len(labels_list) < 2:
        raise ValueError(
            f"Need at least 2 valid seed runs for consensus clustering, "
            f"got {len(labels_list)}. Run more seeds via the sweep first."
        )

    print(f"\n[Consensus] Combining {len(labels_list)} independent seed runs "
          f"into one consensus partition (K={K}) ...")

    # ── Build co-association matrix + final consensus labels ────────────
    C = build_coassociation_matrix(labels_list)
    consensus = consensus_labels_from_coassociation(C, K)

    stability = cluster_stability_scores(C, consensus, K)
    print(f"[Consensus] Per-cluster stability (mean co-association, "
          f"1.0=perfect agreement across all seeds):")
    for k, s in stability.items():
        print(f"    Cluster {k}: {s:.4f}")

    # ── Evaluate consensus labels using the BEST individual seed's Z ────
    best_idx  = int(np.nanargmax(seed_sils))
    best_dir, best_enc, Z_reference = seed_encoders[best_idx]
    print(f"\n[Consensus] Using {best_dir.name} (individual Sil="
          f"{seed_sils[best_idx]:.4f}) as the reference embedding space "
          f"for evaluating the consensus partition.")

    metrics = evaluate_consensus_metrics(Z_reference, consensus)

    print(f"\n{'='*60}")
    print(f"CONSENSUS CLUSTERING RESULTS  [{dataset.upper()}, K={K}, "
          f"{len(labels_list)} seeds]")
    print(f"{'='*60}")
    print(f"  Silhouette Score        = {metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index    = {metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz Index = {metrics['calinski_harabasz']:.2f}")
    print(f"\n  For comparison, individual seed Silhouettes were:")
    for sd_path, sil in zip([s[0] for s in seed_encoders], seed_sils):
        marker = "  <- reference" if sd_path == best_dir else ""
        print(f"    {sd_path.name:<30} Sil={sil:.4f}{marker}")

    # ── Save everything ───────────────────────────────────────────────
    np.save(str(out / f"consensus_labels_K{K}.npy"), consensus)
    np.save(str(out / f"coassociation_matrix_K{K}.npy"), C)

    import pandas as pd
    summary_rows = [{"source": "consensus", "silhouette": metrics["silhouette"],
                     "davies_bouldin": metrics["davies_bouldin"],
                     "calinski_harabasz": metrics["calinski_harabasz"],
                     "n_seeds_combined": len(labels_list)}]
    for sd_path, sil in zip([s[0] for s in seed_encoders], seed_sils):
        summary_rows.append({"source": sd_path.name, "silhouette": sil,
                             "davies_bouldin": np.nan,
                             "calinski_harabasz": np.nan,
                             "n_seeds_combined": np.nan})
    pd.DataFrame(summary_rows).to_csv(
        str(out / f"consensus_summary_K{K}.csv"), index=False)

    print(f"\n[Consensus] Saved:")
    print(f"    {out}/consensus_labels_K{K}.npy")
    print(f"    {out}/coassociation_matrix_K{K}.npy")
    print(f"    {out}/consensus_summary_K{K}.csv")

    return {
        "consensus_labels": consensus,
        "coassociation":    C,
        "metrics":          metrics,
        "stability":        stability,
        "seed_sils":        dict(zip([s[0].name for s in seed_encoders], seed_sils)),
        "reference_seed":   best_dir.name,
    }


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Consensus clustering across seeds")
    p.add_argument("--dataset",    choices=["london", "irish"], required=True)
    p.add_argument("--cache_dir",  required=True)
    p.add_argument("--seed_dirs",  nargs="+", required=True,
                   help="List of result directories, one per seed run "
                        "(each must contain cluster/K<K>/cluster_labels.npy)")
    p.add_argument("--K",          type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_steps",    type=int, default=7)
    p.add_argument("--n_length",   type=int, default=48)
    p.add_argument("--n_features", type=int, default=1)
    p.add_argument("--latent_dim", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_consensus_clustering(
        dataset    = args.dataset,
        cache_dir  = args.cache_dir,
        seed_dirs  = args.seed_dirs,
        K          = args.K,
        output_dir = args.output_dir,
        n_steps    = args.n_steps,
        n_length   = args.n_length,
        n_features = args.n_features,
        latent_dim = args.latent_dim,
        batch_size = args.batch_size,
    )
