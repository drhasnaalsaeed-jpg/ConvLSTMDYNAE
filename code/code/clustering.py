"""
clustering.py
=============
Phase II — Dynamic Deep Embedded Clustering  (Section 5.3, Algorithm 1)

The mathematical methodology is UNCHANGED from the published paper:
Equations 15-25, the encoder/decoder/critic architectures, latent
dimensionality, dropout, kernel sizes, and the Student-t soft-assignment
+ dynamic L1/L2 clustering loss are all identical. Every item below is
an implementation/engineering refinement, not a methodology change.

Base fixes (pre-existing, over the original naive port):
  1. Infinite loop — outer loop now has MaxOuter limit (default 15)
  2. Collapse bug — reinitialise centroids from K-means if collapse detected
  3. L2 weight — explicit L2 scaling to keep compactness loss meaningful
  4. Convergence — exits cleanly when tau_p < total_thresh OR MaxOuter reached

DynAE-faithful additions (ported from the original nairouz/DynAE repo,
`generate_centers()` in DynAE.py):
  5. Centroids are SNAPPED to the nearest real sample's embedding rather
     than left as an abstract K-Means mean — see
     snap_centroids_to_nearest_sample(). Guarantees the "centroid" the
     decoder reconstructs toward is always a point it has genuinely
     learned to decode, not an average sitting in an unexplored region.
  6. Centroid/threshold update schedule — DEFAULT is `always_update_
     centroids=True`, matching this paper's own Fig.6 flowchart (no
     stall-check branch is drawn; "Update centroids" and "Update α1,α2"
     happen every outer iteration unconditionally). The nairouz/DynAE
     repo's event-triggered variant (only regenerate when the conflicted
     count stalls) is available via `always_update_centroids=False` for
     comparison, but is NOT the default — an isolated, controlled A/B
     test (identical seed + pretrained weights, Irish K=3) showed the
     unconditional schedule performs better and is more consistent with
     the paper's own stated rationale for the dynamic loss ("gradual and
     smooth reduction of reconstruction loss").
  7. kappa_min_fraction=0.5 (default) — kappa_min = kappa_init * this.
     A controlled A/B test (same seed/weights) showed 0.5 beats an
     aggressive 0.02 floor on Sil/DBI/CHI simultaneously — an
     over-aggressive floor rushes households out of self-supervision
     too fast, which is exactly the "abrupt removal" failure mode the
     paper's own text warns increases Feature Randomness (FR).
  8. Feature Drift (FD) — optional (`track_feature_drift=True`)
     diagnostic: cosine similarity between grad(L1) and grad(L2) w.r.t.
     shared weights. The repo defines this (`cos_grad` in metrics.py)
     but never wires it up. Feature Randomness (FR), the repo's other
     diagnostic, requires ground-truth labels and is structurally
     impossible for unlabelled household data — not implemented.

==========================================================================
[NEW] Implementation refinements (this revision) — see REFINEMENTS spec
==========================================================================
Every item below is an ENGINEERING refinement only — no equation, no
architecture, no loss function was changed. Each is explained against
what it costs/buys, consistent with REFINEMENTS §15 ("reject
modifications that produce cluster collapse, extremely imbalanced
clusters, or artificially inflated Silhouette; every refinement must
improve robustness across multiple seeds, not chase one seed's number").

  [NEW §2]  Best-clustering-checkpoint tracking (`track_best_checkpoint`,
            default True). Every `checkpoint_monitor_interval` inner
            iterations, Sil/DBI/CHI/cluster-sizes are evaluated on a
            FIXED monitoring subset (not the training batch), combined
            into a COMPOSITE score (normalised Sil + inverted-DBI +
            CHI, averaged — the SAME formula load_profiling.py uses for
            K-selection, via a running min-max normalisation over
            checkpoints seen so far, since future checkpoints aren't
            known yet at decision time), and the encoder+decoder are
            saved if this is the best-so-far COMPOSITE score AND the
            partition passes the balance guard (see §15 below). Raw
            Silhouette alone previously drove this decision; changed on
            explicit request so DBI/CHI can't be silently sacrificed
            for a Sil-only improvement. The FINAL training iteration is
            no longer automatically used — the best checkpoint is
            restored before saving.
            WHY: the final SGD step is not guaranteed to be the best
            one; this removes that specific source of variance without
            touching the optimisation objective itself.

  [NEW §3, clustering]  Optional inner-loop early stopping
            (`clustering_early_stopping`, default False — the outer
            loop's own tau_p convergence check remains the primary,
            already-validated stopping mechanism). When enabled, stops
            the inner loop if the monitoring Sil hasn't improved for
            `clustering_patience` iterations.
            WHY off by default: adding a second, more aggressive
            stopping rule on top of an already-working mechanism risks
            interacting with it in ways that weren't part of the
            controlled tests already run on this pipeline (max_iter=1500
            sweet-spot, kappa_min=0.5 result) — kept opt-in so it can be
            tested in isolation before becoming a default.

  [NEW §6]  K-Means initialisation upgraded from n_init=20 to n_init=50,
            with `init="k-means++"` made explicit everywhere (was
            already sklearn's default, now stated rather than implicit).
            WHY: more restarts + explicit k-means++ seeding reduces the
            chance any single centroid initialisation is a poor local
            optimum — pure initialisation quality, no change to what
            is being optimised.

  [NEW §7]  Unified fixed monitoring subset (seed=42, size=min(4096,N)),
            shared by inner-Silhouette tracking, best-checkpoint
            evaluation, and latent diagnostics — never re-sampled
            within a single cluster() call, so every diagnostic reads
            from the SAME households every time and curves are directly
            point-to-point comparable. Mirrors pretrain.py's existing
            X_mon design for consistency across both phases.

  [NEW §8]  compute_cluster_latent_statistics() — same diagnostic
            vocabulary as pretrain.py's Phase I version (global/per-dim
            variance, active dimensions, active ratio, latent norm/
            mean/std/min/max), now also available during Phase II.
            Warns when active_ratio < 50% (latent collapse signal).
            Visualisation/monitoring only — never fed into any loss.

  [NEW §9]  Comprehensive cluster monitoring — at every checkpoint
            interval, `history` now also records largest/smallest
            cluster percentage and latent active-ratio/global-variance,
            not just Sil/DBI/CHI, giving a fuller picture of partition
            health over training than was previously logged.

  [NEW §13] Reproducibility fix — `seed` now threads through EVERY
            K-Means call inside cluster() (initial centroid
            construction, DynAE-style regeneration, and the
            post-training --polish refinement). Previously,
            `init_centroids_kmeans()` was called without passing
            `random_state` at all, so it silently used the hardcoded
            default of 42 regardless of what `--seed` was actually
            requested — meaning two different `--seed` values produced
            IDENTICAL K-Means initialisation. Centroid-regeneration
            calls during training used only `outer_iter` as the K-Means
            seed, which is identical across every run regardless of
            `--seed` too. Both are now `seed`-dependent, so "same seed
            -> identical results" and "different seed -> genuinely
            different initialisation" both hold as they should.

  [NEW §15] Balance guard on best-checkpoint selection — a checkpoint
            is only eligible to become "best" if its smallest cluster
            is >= `checkpoint_min_cluster_fraction` (default 3%) of N.
            WHY: this is the same failure mode documented in
            load_profiling.py's K-selection filter — a degenerate
            partition (one dominant cluster + tiny fragments) can show
            an artificially inflated Silhouette. Without this guard,
            best-checkpoint tracking could actively reward exactly the
            failure mode REFINEMENTS §15 says to reject.

Note on metrics: the original repo reports supervised acc/nmi/ari
(Hungarian-matched against ground-truth labels — see metrics.py), which
requires known true classes (e.g. MNIST digit labels). Household smart-
meter data has no ground-truth cluster labels, so this pipeline reports
the unsupervised internal validity indices instead: Silhouette,
Davies-Bouldin, and Calinski-Harabasz — the correct substitution for an
unlabelled clustering task, not a simplification.
"""

import numpy as np
import time
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score,
                              davies_bouldin_score,
                              calinski_harabasz_score)
from pathlib import Path


# ===========================================================================
# DynAE-faithful centroid construction (nairouz/DynAE `generate_centers()`)
# ===========================================================================

def snap_centroids_to_nearest_sample(centroids: np.ndarray,
                                     Z:         np.ndarray) -> np.ndarray:
    """
    Mirrors `generate_centers()` in the original nairouz/DynAE repo.

    Rather than using the raw K-Means mean centroid directly, each centroid
    is SNAPPED to the embedding of the nearest real sample (1-NN, ball_tree).
    This guarantees the "centroid" fed to the decoder is always a genuinely
    decodable point the decoder has actually learned to reconstruct from,
    rather than an abstract average that may sit in a latent region the
    decoder has never seen and cannot reconstruct coherently — especially
    important early in clustering, before the latent space is fully shaped.

    Returns the snapped centroids (K, latent_dim), dtype float32.
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Z)
    _, indices = nn.kneighbors(centroids)
    snapped = Z[indices.ravel()]
    return snapped.astype(np.float32)


def kmeans_then_snap(Z: np.ndarray, K: int, random_state: int = 42,
                     n_init: int = 50) -> np.ndarray:
    """K-Means on Z, then snap each resulting centroid to its nearest real
    sample (see snap_centroids_to_nearest_sample). Used both for initial
    centroid construction and for DynAE-style centroid regeneration when
    the conflicted-point count stalls (see the outer loop in cluster())."""
    # [MODIFIED §6] explicit k-means++ (was already sklearn's default, made
    # explicit per spec) + n_init configurable (default 50, up from the
    # original 20), keeping the minimum-inertia run.
    km = KMeans(n_clusters=K, init="k-means++", n_init=n_init,
               random_state=random_state)
    km.fit(Z)
    return snap_centroids_to_nearest_sample(km.cluster_centers_, Z)


# ===========================================================================
# Core equations  (Equations 13–25 from paper)
# ===========================================================================

def soft_assignments(z: tf.Tensor,
                     centroids: tf.Tensor,
                     alpha: float = 1.0) -> tf.Tensor:
    """Soft cluster assignments via Student's t-kernel (Eq. 13)."""
    z_exp   = tf.expand_dims(z, axis=1)
    mu_exp  = tf.expand_dims(centroids, axis=0)
    sq_dist = tf.reduce_sum(tf.square(z_exp - mu_exp), axis=2)
    num     = tf.pow(1.0 + sq_dist / alpha, -(alpha + 1.0) / 2.0)
    denom   = tf.reduce_sum(num, axis=1, keepdims=True) + 1e-9
    return num / denom


def hard_assignments(Q: tf.Tensor) -> tf.Tensor:
    """Argmax cluster assignment sigma(x_i) (Eq. 14)."""
    return tf.argmax(Q, axis=1)


def compute_thresholds(kappa: float, K: int):
    """alpha1 = kappa/K, alpha2 = alpha1/2  (Eq. 21)."""
    alpha1 = kappa / K
    alpha2 = alpha1 / 2.0
    return alpha1, alpha2


def split_conflicted(Q: np.ndarray, alpha1: float, alpha2: float):
    """Conflicted/unconflicted split (Eq. 19, 24)."""
    sorted_Q = np.sort(Q, axis=1)[:, ::-1]
    p1 = sorted_Q[:, 0]
    p2 = sorted_Q[:, 1] if Q.shape[1] > 1 else np.zeros_like(p1)
    conflicted_mask   = (p1 < alpha1) | ((p1 - p2) < alpha2)
    unconflicted_mask = ~conflicted_mask
    return conflicted_mask, unconflicted_mask


def training_progress(n_conflicted: int, N: int) -> float:
    """tau_p = |S_bar| / N  (Eq. 22)."""
    return n_conflicted / max(N, 1)


def dynamic_L1(x_batch:         tf.Tensor,
               x_hat_batch:     tf.Tensor,
               labels:          tf.Tensor,
               centroids:       tf.Tensor,
               decoder:         tf.keras.Model,
               conflicted_mask: np.ndarray) -> tf.Tensor:
    """Dynamic loss L1 (Eq. 20)."""
    recon_loss          = tf.reduce_sum(
        tf.square(x_batch - x_hat_batch), axis=[1, 2, 3, 4])
    centroid_for_sample = tf.gather(centroids, labels)
    x_from_centroid     = decoder(centroid_for_sample, training=True)
    centroid_loss       = tf.reduce_sum(
        tf.square(x_from_centroid - x_hat_batch), axis=[1, 2, 3, 4])
    conflict_tf   = tf.constant(conflicted_mask.astype(np.float32))
    L1_per_sample = conflict_tf * recon_loss + (1.0 - conflict_tf) * centroid_loss
    return tf.reduce_mean(L1_per_sample)


def compactness_L2(z_batch:          tf.Tensor,
                   labels:            tf.Tensor,
                   centroids:         tf.Tensor,
                   unconflicted_mask: np.ndarray) -> tf.Tensor:
    """Compactness loss L2 (Eq. 23)."""
    assigned_centroid = tf.gather(centroids, labels)
    dist_sq           = tf.reduce_sum(
        tf.square(z_batch - assigned_centroid), axis=1)
    unconflict_tf     = tf.constant(unconflicted_mask.astype(np.float32))
    n_unconflicted    = tf.maximum(tf.reduce_sum(unconflict_tf), 1.0)
    return tf.reduce_sum(unconflict_tf * dist_sq) / n_unconflicted


# ===========================================================================
# Feature Drift (FD) — nairouz/DynAE diagnostic (never executed in the
# original repo — the computation was written but left commented out).
#
# FD = cosine similarity between grad(L1) and grad(L2) w.r.t. shared
# encoder/decoder weights. L1 plays the role of the repo's "self-supervised"
# loss (reconstruct yourself if conflicted, reconstruct your centroid if
# confident) and L2 plays the role of "pseudo-supervised" (pull confident
# embeddings toward their centroid). FD answers: are these two loss terms
# pulling the network in agreeing (FD near +1) or conflicting (FD near -1,
# or near 0 for unrelated/orthogonal) directions?
#
# NOTE: unlike Feature Randomness (FR) — which the repo defines against a
# gradient computed from GROUND-TRUTH labels and is therefore structurally
# impossible to compute for unlabelled household data — FD needs no labels
# at all, so it is fully computable here.
# ===========================================================================

def cosine_similarity_flat_grads(grads1, grads2) -> float:
    """
    Cosine similarity between two gradient lists (as returned by
    tf.GradientTape.gradient), flattened and concatenated into single
    vectors. None entries (a variable that received no gradient from one
    of the two losses) are treated as zero for that loss, matching how
    TensorFlow would apply them via apply_gradients.
    """
    v1_parts, v2_parts = [], []
    for g1, g2 in zip(grads1, grads2):
        if g1 is None and g2 is None:
            continue
        # Use whichever gradient is present to get the correct shape for
        # the zero-fill of the missing one.
        ref = g1 if g1 is not None else g2
        a = tf.reshape(g1, [-1]) if g1 is not None else tf.zeros(
            [tf.size(ref)], dtype=ref.dtype)
        b = tf.reshape(g2, [-1]) if g2 is not None else tf.zeros(
            [tf.size(ref)], dtype=ref.dtype)
        v1_parts.append(a)
        v2_parts.append(b)

    if not v1_parts:
        return float("nan")

    v1 = tf.concat(v1_parts, axis=0)
    v2 = tf.concat(v2_parts, axis=0)

    n1 = tf.norm(v1)
    n2 = tf.norm(v2)
    if float(n1) < 1e-12 or float(n2) < 1e-12:
        return float("nan")

    cos_sim = tf.reduce_sum(v1 * v2) / (n1 * n2)
    return float(cos_sim)


# ===========================================================================
# K-means centroid initialisation
# ===========================================================================

def init_centroids_kmeans(encoder:      tf.keras.Model,
                           X:            np.ndarray,
                           K:            int,
                           batch_size:   int = 256,
                           random_state: int = 42,
                           n_init:       int = 50) -> np.ndarray:
    """Encode all samples then run K-means to get initial centroids."""
    print(f"[Cluster Init] Encoding {X.shape[0]} samples ...")
    z_all = []
    for i in range(0, X.shape[0], batch_size):
        z_all.append(encoder(X[i:i + batch_size], training=False).numpy())
    Z = np.vstack(z_all)

    if np.isnan(Z).any():
        print(f"[Cluster Init] WARNING: NaN in embeddings — replacing with 0.")
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[Cluster Init] Running K-means (K={K}, n_init={n_init}) ...")
    km = KMeans(n_clusters=K, init="k-means++", n_init=n_init,
               random_state=random_state)
    km.fit(Z)
    print(f"[Cluster Init] K-means inertia: {km.inertia_:.4f}")

    unique, counts = np.unique(km.labels_, return_counts=True)
    dist = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
    print(f"[Cluster Init] K-means distribution: {dist}")

    # Check balance — retry up to 5 times if any cluster < 5% of total
    N_total = Z.shape[0]
    min_cluster_frac = counts.min() / N_total
    retries = 0
    while min_cluster_frac < 0.05 and retries < 5:
        retries += 1
        print(f"[Cluster Init] Unbalanced (min cluster={counts.min()}, "
              f"{min_cluster_frac*100:.1f}%). Retry {retries}/5 ...")
        km2 = KMeans(n_clusters=K, init="k-means++", n_init=n_init,
                     random_state=random_state + retries * 7)
        km2.fit(Z)
        unique2, counts2 = np.unique(km2.labels_, return_counts=True)
        min_frac2 = counts2.min() / N_total
        if min_frac2 > min_cluster_frac:
            km, unique, counts = km2, unique2, counts2
            min_cluster_frac = min_frac2
            dist = dict(zip([int(u) for u in unique],
                            [int(c) for c in counts]))
            print(f"[Cluster Init] Better distribution found: {dist}")

    return snap_centroids_to_nearest_sample(km.cluster_centers_, Z), Z


# ===========================================================================
# Main clustering loop  (Algorithm 1)
# ===========================================================================

# ===========================================================================
# [NEW — REFINEMENTS §8] Latent diagnostics for the clustering phase
# Mirrors pretrain.py's compute_latent_statistics() so the same diagnostic
# vocabulary (active dims, active ratio, collapse warning) is available
# during Phase II, not just Phase I. Visualization/monitoring only — never
# used inside any loss term or gradient computation.
# ===========================================================================

def compute_cluster_latent_statistics(Z: np.ndarray,
                                      active_threshold: float = 0.01) -> dict:
    """Same computation as pretrain.py's compute_latent_statistics() —
    duplicated locally (rather than cross-imported) to keep clustering.py
    and pretrain.py independent, single-purpose modules."""
    Z       = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    per_dim = np.var(Z, axis=0)
    max_var = per_dim.max() if per_dim.max() > 0 else 1.0
    active  = int((per_dim > active_threshold * max_var).sum())
    norms   = np.linalg.norm(Z, axis=1)
    return {
        "global_var":   float(np.var(Z)),
        "active_dims":  active,
        "active_ratio": active / max(Z.shape[1], 1),
        "latent_norm":  float(norms.mean()),
        "latent_mean":  float(Z.mean()),
        "latent_std":   float(Z.std()),
        "latent_min":   float(Z.min()),
        "latent_max":   float(Z.max()),
    }


def cluster(X:               np.ndarray,
            encoder:         tf.keras.Model,
            decoder:         tf.keras.Model,
            K:               int,
            max_iter:        int   = 1000,     # paper: MaxItr=1000
            max_outer:       int   = 10,
            batch_size:      int   = 256,
            lr:              float = 1e-3,     # paper: SGD lr=0.001
            momentum:        float = 0.9,
            kmeans_n_init:   int   = 50,       # K-Means restarts, used at
                                                # initial centroid construction
                                                # AND every centroid regeneration
                                                # during training. Higher = more
                                                # exhaustive search for the best
                                                # K-Means init, at the cost of
                                                # more compute per regeneration.
            kappa:           float = None,
            kappa_drop_rate: float = 0.3,      # DynAE: kappa *= this
            kappa_min_fraction: float = 0.5,   # kappa_min = kappa_init * this.
                                                # REVERTED to 0.5 after a
                                                # controlled A/B test (same
                                                # seed, same pretrained
                                                # weights, Irish K=3):
                                                #   0.5  -> Sil=0.8449 DBI=0.2031 CHI=64080
                                                #   0.02 -> Sil=0.8094 DBI=0.2540 CHI=48642
                                                # 0.5 won on all 3 metrics.
                                                # Consistent with the paper's
                                                # own stated rationale: "the
                                                # dynamic loss ensures a
                                                # gradual and smooth reduction
                                                # of reconstruction loss" — a
                                                # floor this low rushes too
                                                # many households out of
                                                # self-supervision too fast,
                                                # which is exactly the abrupt-
                                                # removal failure mode the
                                                # paper warns increases FR.
            always_update_centroids: bool = True,  # paper Fig.6 flowchart:
                                                # "Update centroids" and
                                                # "Update α1,α2" happen every
                                                # outer iteration unconditionally
                                                # (no stall-check branch shown).
                                                # Set False to use the
                                                # nairouz/DynAE repo's event-
                                                # triggered variant instead
                                                # (only regenerate on stall).
            total_thresh:    float = 0.01,
            l2_weight:       float = 0.05,     # gentle L2 — guides without collapsing K=3
            pretrained_path: str   = None,
            save_path:       str   = "cluster_weights",
            track_feature_drift: bool = False,  # opt-in — adds ~1 extra
                                                # backward pass per inner
                                                # iteration when enabled
            track_inner_silhouette: bool = False,  # opt-in — periodically
                                                # computes Silhouette DURING
                                                # the inner loop (not just
                                                # once per outer iteration),
                                                # on a fixed sample against
                                                # the current (frozen)
                                                # centroids. Lets you see
                                                # whether Sil improves
                                                # monotonically within an
                                                # outer step's training block,
                                                # or fluctuates/degrades.
            inner_sil_interval: int = 50,      # compute every N inner iters
            inner_sil_sample_size: int = 4096, # [MODIFIED §7] was 2000 —
                                                # aligned to spec cap of
                                                # min(4096,N)
            seed:            int   = 42,       # [NEW §13] reproducibility —
                                                # threads through every
                                                # KMeans call in this
                                                # cluster() invocation
                                                # (previously several were
                                                # hardcoded to 42 regardless
                                                # of the seed actually
                                                # requested, which broke
                                                # "same seed -> identical
                                                # results" for anything
                                                # depending on centroid init)
            track_best_checkpoint: bool = True,  # [NEW §2] every
                                                # checkpoint_monitor_interval
                                                # inner iters, evaluate Sil/
                                                # DBI/CHI/sizes on the FIXED
                                                # monitoring subset; save
                                                # encoder+decoder if this is
                                                # the best-so-far AND passes
                                                # the balance guard (§15);
                                                # restore best (not final)
                                                # weights at the end.
            checkpoint_monitor_interval: int = 100,
            checkpoint_min_cluster_fraction: float = 0.03,  # [NEW §15]
                                                # reject a "best" checkpoint
                                                # if its smallest cluster is
                                                # a degenerate fragment —
                                                # prevents a collapsed
                                                # partition with inflated
                                                # Sil from being saved as
                                                # "best" (see the K=4..10
                                                # degenerate-cluster failure
                                                # mode documented in
                                                # load_profiling.py).
            checkpoint_composite_weights: tuple = (0.4, 0.2, 0.2, 0.2),  # [NEW]
                                                # (sil, dbi, chi, balance)
                                                # weights, must sum to 1.
                                                # Balance = normalised entropy
                                                # of cluster sizes — added
                                                # because the balance GUARD
                                                # (checkpoint_min_cluster_
                                                # fraction) is only a binary
                                                # pass/fail floor, not a
                                                # preference: a checkpoint at
                                                # {3.1%,3.1%,93.8%} clears a
                                                # 3% floor but is still
                                                # heavily skewed. Including
                                                # balance directly in the
                                                # composite means, among
                                                # checkpoints that pass the
                                                # floor, more evenly-sized
                                                # partitions are actively
                                                # preferred, not just
                                                # tolerated.
            checkpoint_min_valid_for_composite: int = 3,  # [NEW] fall back
                                                # to Sil-only comparison
                                                # until at least this many
                                                # balance-guard-passing
                                                # checkpoints have been seen.
                                                # Running min-max normalisation
                                                # is maximally volatile with
                                                # very few points (with only
                                                # 2, one metric being the
                                                # single worst of the pair
                                                # zeroes it out entirely) —
                                                # this avoids composite
                                                # scoring making a confident-
                                                # looking but noise-driven
                                                # decision too early.
            clustering_early_stopping: bool = False,  # [NEW §3] opt-in —
                                                # the outer loop already has
                                                # its own principled
                                                # convergence check (tau_p
                                                # < total_thresh), so this
                                                # is an ADDITIONAL, stricter
                                                # inner-loop-level stop based
                                                # on monitoring Sil plateauing.
                                                # Off by default to avoid
                                                # interfering with the
                                                # already-validated outer
                                                # convergence behaviour.
            clustering_patience: int = 500,    # in inner-iteration units
            verbose:         int   = 20):
    """
    Phase II dynamic clustering (Algorithm 1, Section 5.3.3).

    Key parameters
    --------------
    max_iter        : inner loop iterations per outer step (paper: ~100)
    max_outer       : maximum outer iterations before forced exit (default 10)
    lr              : SGD learning rate (reduced to 1e-4 to prevent collapse)
    kappa           : confidence threshold (default 0.3*K, paper Sec 6.2.2)
    kappa_drop_rate : DynAE-faithful decay factor — on each event-triggered
                     regeneration (conflicted count stalled/increased),
                     kappa *= kappa_drop_rate (direct multiply, matching
                     nairouz/DynAE's `kappa = 0.3 * kappa`). NOT applied
                     every outer iteration — only on the stall trigger.
    l2_weight       : scale factor for L2 compactness loss (keeps it meaningful)
    total_thresh    : convergence: tau_p < this fraction → stop (0.01 = 1%)
    track_inner_silhouette : if True, computes Silhouette every
                     `inner_sil_interval` inner iterations on a fixed
                     `inner_sil_sample_size`-household sample, evaluated
                     against the CURRENT (frozen-for-this-outer-step)
                     centroids. Adds encode+silhouette cost at that
                     interval — keep the sample small for speed.
    track_feature_drift : if True, logs Feature Drift (FD) — cosine
                     similarity between grad(L1) and grad(L2) w.r.t. shared
                     weights, at each `verbose` interval. FD near +1 means
                     the reconstruction and compactness objectives are
                     pulling the network in agreeing directions (healthy);
                     near -1 means they are actively conflicting. Adds a
                     second backward pass per inner iteration when enabled
                     (see cosine_similarity_flat_grads).
    """
    t_cluster_start = time.time()

    if pretrained_path:
        p = Path(pretrained_path)
        encoder.load_weights(str(p / "encoder.weights.h5"))
        decoder.load_weights(str(p / "decoder.weights.h5"))
        print(f"[Cluster] Pretrained weights loaded from {pretrained_path}")

    N = X.shape[0]

    # [MODIFIED §7] Unified fixed monitoring subset — shared by inner-
    # Silhouette tracking, best-checkpoint evaluation, and latent
    # diagnostics, so every diagnostic reads from the SAME households
    # every time (never re-sampled), matching pretrain.py's X_mon design.
    need_monitoring_subset = track_inner_silhouette or track_best_checkpoint
    if need_monitoring_subset:
        mon_size = min(inner_sil_sample_size, N)
        _mon_rng = np.random.default_rng(42)
        mon_idx  = _mon_rng.choice(N, size=mon_size, replace=False)
        X_sil_sample = X[mon_idx].astype(np.float32)
        print(f"[Cluster] Monitoring subset: {mon_size} samples "
              f"(fixed, seed=42) — used for Sil/DBI/CHI/sizes/latent-stats")
        if track_inner_silhouette:
            print(f"[Cluster] Inner-iteration Silhouette tracking enabled — "
                  f"every {inner_sil_interval} iters")
        if track_best_checkpoint:
            print(f"[Cluster] Best-checkpoint tracking enabled — every "
                  f"{checkpoint_monitor_interval} iters, min_cluster_frac"
                  f">={checkpoint_min_cluster_fraction}")

    # [NEW §2] Best-clustering-checkpoint state
    # [MODIFIED] Selection now driven by a composite score (normalised
    # Sil + inverted-DBI + CHI, averaged — same formula as load_profiling.py's
    # K-selection), not raw Silhouette alone.
    best_monitor_composite = -1.0  # -1 is below any valid composite value ([0,1] range)
    best_monitor_sil    = -1.0   # kept for reporting/back-compat only —
                                  # no longer the selection criterion itself
    best_monitor_iter   = 0
    best_monitor_metrics = None
    checkpoint_sp = None
    if track_best_checkpoint:
        checkpoint_sp = Path(save_path)
        checkpoint_sp.mkdir(parents=True, exist_ok=True)

    # [NEW §3, clustering] Optional inner-loop early-stopping state
    monitor_no_improve = 0

    kappa_init = 0.3 * K if kappa is None else kappa
    kappa      = kappa_init
    # BUG FIX: kappa_init/2.0 was tuned for an older, gentler decay formula
    # (kappa *= (1 - drop_rate), i.e. x0.7/step). With the corrected
    # kappa_drop_rate semantics (direct multiply, kappa *= drop_rate,
    # i.e. x0.3/step, matching the paper), that floor is HIGHER than the
    # very first decay step (0.9*0.3=0.27 < 0.45), so kappa clamped to the
    # floor on outer iteration 1 and never moved again — the paper's
    # progressive confidence-threshold tightening (Fig.6) was silently
    # disabled the whole time. A much lower floor lets kappa actually
    # decay through several genuine stages before settling.
    kappa_min  = kappa_init * kappa_min_fraction

    # Initialise centroids with K-means, snapped to nearest real sample
    # (DynAE-faithful — see snap_centroids_to_nearest_sample)
    centroids, Z_init = init_centroids_kmeans(encoder, X, K, batch_size,
                                              random_state=seed,
                                              n_init=kmeans_n_init)
    centroids_var = tf.Variable(centroids, trainable=False, dtype=tf.float32)

    alpha1, alpha2 = compute_thresholds(kappa, K)
    print(f"[Cluster] K={K}  κ={kappa:.3f}  κ_min={kappa_min:.3f}  "
          f"α₁={alpha1:.4f}  α₂={alpha2:.4f}  "
          f"lr={lr}  max_iter={max_iter}  max_outer={max_outer}")

    optimiser = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (np.arange(N), X.astype(np.float32))
        )
        .shuffle(buffer_size=min(N, 5000), seed=42)
        .batch(batch_size, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    dataset_iter = iter(dataset)

    history = {
        "L": [], "L1": [], "L2": [],
        "silhouette": [], "davies_bouldin": [], "calinski_harabasz": [],
        "tau_p": [], "n_conflicted": [], "N_total": N,
        "FD": [],   # Feature Drift — aligned 1:1 with L/L1/L2 by index
                   # when track_feature_drift=True; empty otherwise.
        "inner_sil": [], "inner_sil_iter": [],   # Silhouette DURING the
                   # inner loop (sparse — only at inner_sil_interval steps),
                   # only populated when track_inner_silhouette=True.
        "outer_boundaries": [],   # global_inner_iter value at the START of
                   # each outer iteration's inner loop — lets plots mark
                   # where centroids/thresholds were last updated.
        # [NEW §9] Comprehensive cluster monitoring — populated at
        # checkpoint_monitor_interval whenever track_best_checkpoint=True.
        "monitor_iter":   [], "monitor_sil": [], "monitor_dbi": [],
        "monitor_chi":    [], "monitor_balance": [],  # [NEW] normalised
                   # entropy of cluster sizes — see balance_score above.
        "monitor_largest_pct": [], "monitor_smallest_pct": [],
        "monitor_latent_active_ratio": [], "monitor_latent_global_var": [],
        # [NEW §2] Best-checkpoint bookkeeping — filled in after training.
        "best_checkpoint_iter": None, "best_checkpoint_sil": None,
        "best_checkpoint_composite": None,
        "best_checkpoint_used": False,
    }

    # Global inner-iteration counter — increments across the WHOLE cluster()
    # call (not reset per outer iteration), matching how L/L1/L2 accumulate.
    # Used as the x-axis for the inner-Silhouette plot.
    _global_inner_iter = 0

    # Encode full dataset
    z_full = np.vstack([
        encoder(X[i:i + batch_size], training=False).numpy()
        for i in range(0, N, batch_size)
    ])
    Q_full = soft_assignments(
        tf.constant(z_full, dtype=tf.float32), centroids_var
    ).numpy()

    # n_conflicted_prev starts at N (nairouz/DynAE: nb_conf_prev = x.shape[0]) —
    # i.e. "everything was conflicted before we started", so the very first
    # check can legitimately trigger a regeneration if confidence hasn't
    # improved at all yet.
    n_conflicted_prev = N

    # ── Outer loop ────────────────────────────────────────────────────────
    for outer_iter in range(1, max_outer + 1):

        conflicted_mask, unconflicted_mask = split_conflicted(
            Q_full, alpha1, alpha2
        )
        n_conflicted = int(conflicted_mask.sum())
        tau_p        = training_progress(n_conflicted, N)

        history["tau_p"].append(tau_p)
        history["n_conflicted"].append(n_conflicted)

        # ── Centroid regeneration + threshold decay ─────────────────────────
        # Two modes, controlled by always_update_centroids:
        #
        #   True (default) — paper Fig.6 flowchart: "Update centroids" and
        #     "Update α1,α2" happen every outer iteration unconditionally,
        #     as long as conflicted points remain. No stall-check branch
        #     is shown in the paper's own algorithm diagram.
        #
        #   False — nairouz/DynAE repo's train_dynAE: only regenerate when
        #     nb_conf >= nb_conf_prev (conflicted count failed to improve).
        #     This is the GENERIC DynAE repo's behavior, not necessarily
        #     what this specific paper's Fig.6 documents — kept available
        #     for comparison since the two sources genuinely disagree here.
        should_update = always_update_centroids or (n_conflicted >= n_conflicted_prev)

        if should_update:
            trigger_reason = ("every iteration (paper Fig.6)"
                             if always_update_centroids else
                             f"stalled/increased ({n_conflicted} vs "
                             f"prev {n_conflicted_prev}, DynAE-repo event trigger)")
            print(f"[Cluster] Updating centroids + thresholds — {trigger_reason}")
            z_safe = np.nan_to_num(z_full, nan=0.0, posinf=0.0, neginf=0.0)
            new_centroids = kmeans_then_snap(z_safe, K, random_state=seed + outer_iter * 1000, n_init=kmeans_n_init)
            centroids_var.assign(new_centroids)

            kappa  = max(kappa * kappa_drop_rate, kappa_min)
            alpha1, alpha2 = compute_thresholds(kappa, K)

            Q_full = soft_assignments(
                tf.constant(z_full, dtype=tf.float32), centroids_var
            ).numpy()
            conflicted_mask, unconflicted_mask = split_conflicted(
                Q_full, alpha1, alpha2
            )
            print(f"[Cluster] New thresholds: κ={kappa:.3f}  "
                  f"α₁={alpha1:.4f}  α₂={alpha2:.4f}")
        else:
            print(f"[Cluster] Confidence improving on its own "
                  f"({n_conflicted} < prev {n_conflicted_prev}) — "
                  f"keeping current centroids/thresholds.")

        n_conflicted_prev = n_conflicted

        labels_full  = np.argmax(Q_full, axis=1)
        unique, cnts = np.unique(labels_full, return_counts=True)
        dist_str     = ", ".join([f"C{u}:{c}" for u, c in zip(unique, cnts)])

        print(f"\n[Cluster] Outer iter {outer_iter}/{max_outer} | "
              f"Conflicted: {n_conflicted}/{N} ({tau_p*100:.1f}%) | "
              f"α₁={alpha1:.4f} α₂={alpha2:.4f}")
        print(f"[Cluster] Distribution: {dist_str}")

        # Convergence check — require at least 2 outer iters AND real clusters
        n_active = len(np.unique(np.argmax(Q_full, axis=1)))
        if tau_p < total_thresh and outer_iter >= 2 and n_active >= 2:
            print(f"[Cluster] Converged (τp={tau_p:.4f} < {total_thresh}, "
                  f"outer={outer_iter}, clusters={n_active})")
            break
        elif tau_p < total_thresh and (outer_iter < 2 or n_active < 2):
            print(f"[Cluster] tau_p={tau_p:.4f} < threshold but "
                  f"outer_iter={outer_iter} or clusters={n_active} < K — "
                  "continuing...")

        # Emergency safety net: if a cluster has fully collapsed (empty),
        # reinitialise regardless of the stall-trigger above.
        if len(unique) < K:
            print(f"[Cluster] Collapse detected — reinitialising centroids ...")
            z_safe = np.nan_to_num(z_full, nan=0.0, posinf=0.0, neginf=0.0)
            new_centroids = kmeans_then_snap(z_safe, K, random_state=seed + outer_iter * 1000, n_init=kmeans_n_init)
            centroids_var.assign(new_centroids)
            Q_full = soft_assignments(
                tf.constant(z_full, dtype=tf.float32), centroids_var
            ).numpy()
            conflicted_mask, unconflicted_mask = split_conflicted(
                Q_full, alpha1, alpha2
            )
            print(f"[Cluster] Centroids reinitialised.")

        # ── Inner loop ────────────────────────────────────────────────────
        history["outer_boundaries"].append(_global_inner_iter)
        for inner_it in range(1, max_iter + 1):
            idx_batch, x_batch = next(dataset_iter)
            idx_np = idx_batch.numpy()

            batch_conflicted   = conflicted_mask[idx_np]
            batch_unconflicted = unconflicted_mask[idx_np]

            ae_vars = encoder.trainable_variables + decoder.trainable_variables

            if track_feature_drift:
                # Persistent tape: two separate backward passes, one for
                # L1 and one for L2, so we can measure whether they pull
                # the shared weights in agreeing or conflicting directions
                # (Feature Drift). The combined update gradient is then
                # reconstructed as grad(L1) + l2_weight*grad(L2), which is
                # mathematically identical to grad(L1 + l2_weight*L2) by
                # linearity of differentiation — no third backward pass
                # needed for the actual optimiser step.
                with tf.GradientTape(persistent=True) as tape:
                    z_batch  = encoder(x_batch, training=True)
                    x_hat    = decoder(z_batch, training=True)
                    Q_batch  = soft_assignments(z_batch, centroids_var)
                    labels_b = tf.cast(hard_assignments(Q_batch), dtype=tf.int32)

                    L1 = dynamic_L1(x_batch, x_hat, labels_b, centroids_var,
                                     decoder, batch_conflicted)
                    if l2_weight > 0:
                        L2 = compactness_L2(z_batch, labels_b, centroids_var,
                                            batch_unconflicted)
                    else:
                        L2 = tf.constant(0.0)

                grads_L1 = tape.gradient(L1, ae_vars)
                grads_L2 = (tape.gradient(L2, ae_vars) if l2_weight > 0
                           else [None] * len(ae_vars))
                del tape

                fd = cosine_similarity_flat_grads(grads_L1, grads_L2)
                history["FD"].append(fd)

                def _zero_if_none(g, ref_var):
                    return g if g is not None else tf.zeros_like(ref_var)

                grads = [
                    _zero_if_none(g1, v) + l2_weight * _zero_if_none(g2, v)
                    for g1, g2, v in zip(grads_L1, grads_L2, ae_vars)
                ]
                L = L1 + l2_weight * L2

            else:
                with tf.GradientTape() as tape:
                    z_batch  = encoder(x_batch, training=True)
                    x_hat    = decoder(z_batch, training=True)
                    Q_batch  = soft_assignments(z_batch, centroids_var)
                    labels_b = tf.cast(hard_assignments(Q_batch), dtype=tf.int32)

                    L1 = dynamic_L1(x_batch, x_hat, labels_b, centroids_var,
                                     decoder, batch_conflicted)
                    if l2_weight > 0:
                        L2 = compactness_L2(z_batch, labels_b, centroids_var,
                                            batch_unconflicted)
                        L  = L1 + l2_weight * L2
                    else:
                        L2 = tf.constant(0.0)
                        L  = L1

                grads = tape.gradient(L, ae_vars)

            # Clip gradients to prevent L1 explosion in later outer iterations
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimiser.apply_gradients(zip(grads, ae_vars))

            if inner_it % verbose == 0:
                fd_str = (f"  FD={history['FD'][-1]:.4f}"
                          if track_feature_drift and history["FD"] else "")
                print(f"    inner {inner_it:3d}/{max_iter} | "
                      f"L={L:.4f}  L1={L1:.4f}  "
                      f"L2_scaled={l2_weight*L2:.4f}  L2_raw={L2:.6f}{fd_str}")

            history["L"].append(float(L))
            history["L1"].append(float(L1))
            history["L2"].append(float(L2))
            _global_inner_iter += 1

            # ── Periodic inner-iteration Silhouette (opt-in) ────────────────
            if track_inner_silhouette and (inner_it % inner_sil_interval == 0):
                z_sample = encoder(X_sil_sample, training=False).numpy()
                z_sample = np.nan_to_num(z_sample, nan=0.0)
                Q_sample = soft_assignments(
                    tf.constant(z_sample, dtype=tf.float32), centroids_var
                ).numpy()
                labels_sample = np.argmax(Q_sample, axis=1)
                n_unique = len(np.unique(labels_sample))
                if n_unique >= 2:
                    try:
                        sil_val = float(silhouette_score(
                            z_sample, labels_sample, metric="euclidean"))
                    except Exception:
                        sil_val = np.nan
                else:
                    sil_val = np.nan
                history["inner_sil"].append(sil_val)
                history["inner_sil_iter"].append(_global_inner_iter)
                if inner_it % verbose == 0:
                    sil_str = f"{sil_val:.4f}" if not np.isnan(sil_val) else "NaN"
                    print(f"      [inner Sil] iter {_global_inner_iter}: "
                          f"Sil={sil_str} (sample={len(X_sil_sample)})")

            # ── [NEW §2, §9] Comprehensive best-checkpoint monitoring ────────
            # Every checkpoint_monitor_interval iters: compute Sil/DBI/CHI/
            # sizes/latent-stats on the SAME fixed monitoring subset used
            # above, then save encoder+decoder as the new "best" checkpoint
            # IF Silhouette improved AND the partition passes the balance
            # guard (§15 — never let a degenerate collapse masquerade as
            # a high-Sil "best" state; see load_profiling.py's identical
            # degenerate-cluster filter for the K-selection stage).
            if track_best_checkpoint and (inner_it % checkpoint_monitor_interval == 0):
                z_ckpt = encoder(X_sil_sample, training=False).numpy()
                z_ckpt = np.nan_to_num(z_ckpt, nan=0.0)
                Q_ckpt = soft_assignments(
                    tf.constant(z_ckpt, dtype=tf.float32), centroids_var
                ).numpy()
                labels_ckpt = np.argmax(Q_ckpt, axis=1)
                unique_ckpt, counts_ckpt = np.unique(labels_ckpt, return_counts=True)
                n_unique_ckpt = len(unique_ckpt)

                if n_unique_ckpt >= 2:
                    try:
                        ckpt_sil = float(silhouette_score(
                            z_ckpt, labels_ckpt, metric="euclidean"))
                        ckpt_dbi = float(davies_bouldin_score(z_ckpt, labels_ckpt))
                        ckpt_chi = float(calinski_harabasz_score(z_ckpt, labels_ckpt))
                    except Exception:
                        ckpt_sil, ckpt_dbi, ckpt_chi = np.nan, np.nan, np.nan
                    largest_pct  = float(counts_ckpt.max()) / len(labels_ckpt)
                    smallest_pct = float(counts_ckpt.min()) / len(labels_ckpt)
                    # [NEW] Balance = normalised Shannon entropy of the
                    # cluster-size distribution, over ALL K clusters — not
                    # just the smallest one. A checkpoint at {3.1%,3.1%,
                    # 93.8%} clears the min-cluster-fraction FLOOR but is
                    # still heavily skewed; entropy correctly scores that
                    # as poorly balanced (~0.24), whereas the floor alone
                    # would have silently accepted it. 1.0 = perfectly
                    # even sizes, ->0 = one cluster dominates completely.
                    props = counts_ckpt / counts_ckpt.sum()
                    raw_entropy = float(-(props * np.log(props)).sum())
                    max_entropy = np.log(K) if K > 1 else 1.0
                    balance_score = raw_entropy / max_entropy if max_entropy > 0 else 0.0
                else:
                    ckpt_sil, ckpt_dbi, ckpt_chi = np.nan, np.nan, np.nan
                    largest_pct, smallest_pct = 1.0, 0.0
                    balance_score = 0.0

                lat_stats = compute_cluster_latent_statistics(z_ckpt)

                history["monitor_iter"].append(_global_inner_iter)
                history["monitor_sil"].append(ckpt_sil)
                history["monitor_dbi"].append(ckpt_dbi)
                history["monitor_chi"].append(ckpt_chi)
                history["monitor_balance"].append(balance_score)
                history["monitor_largest_pct"].append(largest_pct)
                history["monitor_smallest_pct"].append(smallest_pct)
                history["monitor_latent_active_ratio"].append(lat_stats["active_ratio"])
                history["monitor_latent_global_var"].append(lat_stats["global_var"])

                if lat_stats["active_ratio"] < 0.5:
                    print(f"  [WARNING] Latent collapse detected during "
                          f"clustering: active_ratio="
                          f"{lat_stats['active_ratio']:.1%} < 50% at iter "
                          f"{_global_inner_iter}")

                if inner_it % verbose == 0:
                    sil_s = f"{ckpt_sil:.4f}" if not np.isnan(ckpt_sil) else "NaN"
                    dbi_s = f"{ckpt_dbi:.4f}" if not np.isnan(ckpt_dbi) else "NaN"
                    chi_s = f"{ckpt_chi:.1f}"  if not np.isnan(ckpt_chi) else "NaN"
                    print(f"      [checkpoint] iter {_global_inner_iter}: "
                          f"Sil={sil_s}  DBI={dbi_s}  CHI={chi_s}  "
                          f"largest={largest_pct:.1%}  "
                          f"smallest={smallest_pct:.1%}  "
                          f"balance={balance_score:.3f}  "
                          f"active_dims={lat_stats['active_dims']}")

                # [MODIFIED] Weighted composite score instead of raw
                # Silhouette alone, but with a volatility guard: falls
                # back to pure Sil-only comparison until
                # checkpoint_min_valid_for_composite balance-guard-passing
                # checkpoints have been seen. Running min-max normalisation
                # with very few points is extremely sensitive — with only
                # 2 valid checkpoints, whichever is worse on ANY single
                # metric gets normalised to exactly 0 for that metric,
                # which can veto a large Sil gain over a tiny CHI/DBI
                # wobble. Sil is also weighted higher (default) than
                # DBI/CHI once composite scoring does kick in, for the
                # same reason. Balance (entropy of cluster sizes) is used
                # DIRECTLY, not running-min-max normalised like the other
                # three — unlike Sil/DBI/CHI, its "good" value (1.0 =
                # perfectly even sizes) is a fixed, absolute ceiling, not
                # something that only makes sense relative to what's been
                # seen so far.
                valid_sils = [v for v in history["monitor_sil"] if not np.isnan(v)]
                valid_dbis = [v for v in history["monitor_dbi"] if not np.isnan(v)]
                valid_chis = [v for v in history["monitor_chi"] if not np.isnan(v)]
                n_valid_so_far = len(valid_sils)  # includes this checkpoint,
                                                  # since it was already
                                                  # appended to history above

                def _running_norm(value, values, higher_better):
                    if np.isnan(value) or len(values) == 0:
                        return np.nan
                    lo, hi = min(values), max(values)
                    if hi - lo < 1e-12:
                        return 1.0  # first point, or no variation yet
                    n = (value - lo) / (hi - lo)
                    return n if higher_better else 1.0 - n

                def _composite_from_raw(sil, dbi, chi, bal, sils, dbis, chis):
                    """Computes a composite score for ANY (sil,dbi,chi,bal)
                    tuple against the GIVEN running min/max lists. Used both
                    for the new checkpoint and to re-score the current best
                    under the SAME (latest) normalisation — see the bug fix
                    below."""
                    s_n = _running_norm(sil, sils, higher_better=True)
                    d_n = _running_norm(dbi, dbis, higher_better=False)
                    c_n = _running_norm(chi, chis, higher_better=True)
                    w_s, w_d, w_c, w_b = checkpoint_composite_weights
                    parts = [(s_n, w_s), (d_n, w_d), (c_n, w_c), (bal, w_b)]
                    valid = [(v, w) for v, w in parts if not np.isnan(v)]
                    return (sum(v * w for v, w in valid) /
                           sum(w for _, w in valid) if valid else np.nan)

                use_composite = n_valid_so_far >= checkpoint_min_valid_for_composite

                if use_composite:
                    composite = _composite_from_raw(
                        ckpt_sil, ckpt_dbi, ckpt_chi, balance_score,
                        valid_sils, valid_dbis, valid_chis)
                    if inner_it % verbose == 0 and not np.isnan(composite):
                        print(f"      [checkpoint] composite={composite:.4f}")
                else:
                    # Not enough valid checkpoints yet — pure Sil-only
                    # comparison, same as the original pre-composite logic.
                    composite = ckpt_sil
                    if inner_it % verbose == 0 and not np.isnan(composite):
                        print(f"      [checkpoint] Sil-only comparison "
                              f"(only {n_valid_so_far}/"
                              f"{checkpoint_min_valid_for_composite} valid "
                              f"checkpoints so far) — Sil={composite:.4f}")

                passes_balance_guard = (
                    n_unique_ckpt >= 2
                    and smallest_pct >= checkpoint_min_cluster_fraction
                )

                # [BUG FIX] The comparison target used to be a FROZEN
                # composite value, recorded once when a checkpoint first
                # won and never updated. Running min-max normalisation
                # means the range widens as training progresses, so a
                # checkpoint compared under an early, narrow range could
                # score higher than one compared later under a wider
                # range — even if the later checkpoint is strictly better
                # on every raw metric. Concrete case this produced: a
                # checkpoint at iter=300 with Sil=0.678 stayed "best" for
                # the rest of a 1500-iteration run, outranking iter=1500
                # (Sil=0.809, better DBI AND CHI too) purely because
                # iter=300 happened to be evaluated when only 3 points
                # existed. Fixed by re-scoring the CURRENT best's stored
                # raw metrics under the SAME (latest) normalisation as the
                # new checkpoint before comparing — an apples-to-apples
                # comparison every time, using only 4 stored numbers, not
                # the old checkpoint's weights.
                if use_composite and best_monitor_metrics is not None:
                    best_composite_now = _composite_from_raw(
                        best_monitor_metrics["sil"], best_monitor_metrics["dbi"],
                        best_monitor_metrics["chi"], best_monitor_metrics["balance"],
                        valid_sils, valid_dbis, valid_chis)
                else:
                    best_composite_now = best_monitor_composite

                # NOTE: >= (not strict >) is intentional here. Running
                # min-max normalisation means a checkpoint that is
                # simultaneously the best-seen-so-far on all 3 metrics
                # normalises to composite=1.0 — including, trivially, the
                # very FIRST checkpoint (a one-point set is its own min
                # and max). With strict >, no later point could ever
                # exceed that first 1.0, so the first checkpoint would
                # always "win" by construction — defeating the purpose
                # entirely. >= lets the most RECENT checkpoint that is
                # still non-dominated on all 3 metrics keep taking over,
                # which is what "best so far" should actually mean.
                if (passes_balance_guard and not np.isnan(composite)
                        and composite >= best_composite_now):
                    best_monitor_composite = composite
                    best_monitor_sil        = ckpt_sil    # kept for reporting/back-compat
                    best_monitor_iter       = _global_inner_iter
                    best_monitor_metrics = {
                        "sil": ckpt_sil, "dbi": ckpt_dbi, "chi": ckpt_chi,
                        "composite": composite, "balance": balance_score,
                        "largest_pct": largest_pct, "smallest_pct": smallest_pct,
                    }
                    encoder.save_weights(
                        str(checkpoint_sp / "encoder_best_cluster.weights.h5"))
                    decoder.save_weights(
                        str(checkpoint_sp / "decoder_best_cluster.weights.h5"))
                    np.save(str(checkpoint_sp / "centroids_best_cluster.npy"),
                           centroids_var.numpy())
                    monitor_no_improve = 0
                    if inner_it % verbose == 0:
                        print(f"      [checkpoint] New best "
                              f"(composite={composite:.4f}, Sil={ckpt_sil:.4f}) — saved.")
                else:
                    monitor_no_improve += 1

                # [NEW §3, clustering] Optional inner-loop early stopping —
                # off by default; the outer loop's tau_p convergence check
                # is the primary/validated stopping mechanism.
                if (clustering_early_stopping
                        and monitor_no_improve * checkpoint_monitor_interval
                            >= clustering_patience):
                    print(f"  [EarlyStopping-cluster] No monitoring "
                          f"improvement for {clustering_patience} iters "
                          f"(best Sil={best_monitor_sil:.4f} at iter "
                          f"{best_monitor_iter}). Stopping inner loop early.")
                    break

        # ── Re-encode for the NEXT outer iteration's stall check ────────────
        # Centroids/thresholds are NOT modified here — that only happens at
        # the top of the next outer iteration, and only if the conflicted
        # count has stalled (see the event-triggered block above). This
        # matches nairouz/DynAE's train_dynAE, where centers are fixed for
        # the whole batch-training block and only regenerated at the next
        # validate_interval check if progress has stalled.
        z_full = np.vstack([
            encoder(X[i:i + batch_size], training=False).numpy()
            for i in range(0, N, batch_size)
        ])
        if np.isnan(z_full).any():
            c_mean = np.nanmean(centroids_var.numpy(), axis=0)
            nan_rows = np.where(np.isnan(z_full).any(axis=1))[0]
            z_full[nan_rows] = c_mean
            z_full = np.nan_to_num(z_full, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  [Cluster] NaN guard: fixed {len(nan_rows)} embeddings.")

        Q_full = soft_assignments(
            tf.constant(z_full, dtype=tf.float32), centroids_var
        ).numpy()
        labels_full = np.argmax(Q_full, axis=1)

        # [NEW — FIGURES spec] Save a snapshot of labels at the end of
        # every outer iteration. Purely additive bookkeeping — does not
        # affect any computation, gradient, or decision in the algorithm.
        # Enables plot_assignment_stability() to show how cluster
        # membership evolved across outer iterations without needing to
        # re-run training.
        if "label_history" not in history:
            history["label_history"] = []
        history["label_history"].append(labels_full.copy())

        # Compute metrics
        m = evaluate_metrics(encoder, X, labels_full, batch_size)
        history["silhouette"].append(m["silhouette"])
        history["davies_bouldin"].append(m["davies_bouldin"])
        history["calinski_harabasz"].append(m["calinski_harabasz"])

        def _fmt(v, d=4):
            return f"{v:.{d}f}" if not np.isnan(v) else "NaN"

        print(f"[Cluster] Metrics: Sil={_fmt(m['silhouette'])}  "
              f"DBI={_fmt(m['davies_bouldin'])}  "
              f"CHI={_fmt(m['calinski_harabasz'], 2)}")

    else:
        print(f"[Cluster] Max outer iterations ({max_outer}) reached.")

    # ── [NEW §2] Restore best clustering checkpoint (not final iteration) ──
    # Matches REFINEMENTS §2 exactly: "Do NOT automatically use the final
    # [...] weights [...] Restore the best checkpoint instead of the final
    # iteration." Only restores if a valid (balance-guard-passing) best
    # checkpoint was actually found during training; otherwise the final
    # training state is used as-is (identical to pre-refinement behaviour).
    if (track_best_checkpoint and best_monitor_metrics is not None
            and checkpoint_sp is not None):
        best_enc_path = checkpoint_sp / "encoder_best_cluster.weights.h5"
        best_dec_path = checkpoint_sp / "decoder_best_cluster.weights.h5"
        best_cent_path = checkpoint_sp / "centroids_best_cluster.npy"
        if best_enc_path.exists():
            print(f"\n[Cluster] Restoring best checkpoint "
                  f"(iter {best_monitor_iter}, composite="
                  f"{best_monitor_composite:.4f}, Sil={best_monitor_sil:.4f}) "
                  f"— was final-iteration state better or worse? Final "
                  f"outer-iteration Sil={history['silhouette'][-1] if history['silhouette'] else float('nan'):.4f}")
            encoder.load_weights(str(best_enc_path))
            decoder.load_weights(str(best_dec_path))
            if best_cent_path.exists():
                centroids_var.assign(np.load(str(best_cent_path)))
            # Re-encode the FULL dataset with the restored (best) encoder
            # so the final saved labels/artifacts reflect the best
            # checkpoint, not the last training step.
            z_full = np.vstack([
                encoder(X[i:i + batch_size], training=False).numpy()
                for i in range(0, N, batch_size)
            ])
            z_full = np.nan_to_num(z_full, nan=0.0, posinf=0.0, neginf=0.0)
            Q_full = soft_assignments(
                tf.constant(z_full, dtype=tf.float32), centroids_var
            ).numpy()
            history["best_checkpoint_iter"] = best_monitor_iter
            history["best_checkpoint_sil"]  = best_monitor_sil
            history["best_checkpoint_composite"] = best_monitor_composite
            history["best_checkpoint_used"] = True
        else:
            print(f"[Cluster] WARNING: best checkpoint files missing — "
                  f"keeping final-iteration weights.")

    # Final labels
    cluster_labels = np.argmax(Q_full, axis=1)
    unique, cnts   = np.unique(cluster_labels, return_counts=True)
    print(f"\n[Cluster] Final distribution: "
          f"{dict(zip([int(u) for u in unique],[int(c) for c in cnts]))}")

    sp = Path(save_path)
    sp.mkdir(parents=True, exist_ok=True)
    encoder.save_weights(str(sp / "encoder_clustered.weights.h5"))
    decoder.save_weights(str(sp / "decoder_clustered.weights.h5"))
    np.save(str(sp / "centroids.npy"), centroids_var.numpy())
    np.save(str(sp / "cluster_labels.npy"), cluster_labels)
    if "label_history" in history and history["label_history"]:
        np.save(str(sp / "label_history.npy"),
               np.array(history["label_history"]))
    print(f"[Cluster] Saved to {sp}/")

    elapsed = time.time() - t_cluster_start
    history["training_time_sec"] = elapsed
    print(f"[Cluster] Clustering time (K={K}): {elapsed:.1f}s "
          f"({elapsed/60:.2f} min)")

    return cluster_labels, history, centroids_var.numpy()


# ===========================================================================
# Three-metric evaluation
# ===========================================================================

def encode_full(encoder, X, batch_size=256):
    parts = []
    for i in range(0, X.shape[0], batch_size):
        parts.append(encoder(X[i:i + batch_size], training=False).numpy())
    return np.vstack(parts)


def evaluate_metrics(encoder, X, cluster_labels, batch_size=256):
    """Silhouette, Davies-Bouldin, Calinski-Harabasz in latent space."""
    _NAN = {"silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan}

    Z = encode_full(encoder, X, batch_size)
    N = Z.shape[0]

    if np.isnan(Z).any():
        print("  [Metrics] WARNING: NaN in embeddings.")
        return _NAN

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    n_unique = len(unique_labels)

    if n_unique < 2:
        print(f"  [Metrics] WARNING: Only {n_unique} cluster.")
        return _NAN
    if counts.min() < 2:
        print(f"  [Metrics] WARNING: Smallest cluster has {counts.min()} sample.")
        return _NAN
    if N <= n_unique:
        print(f"  [Metrics] WARNING: N={N} <= K={n_unique}.")
        return _NAN

    # [FIX] Previously printed "K={n_unique}" — ambiguous with the
    # REQUESTED K when they diverge (i.e. when some requested clusters
    # ended up completely empty). Now explicit about which is which, so
    # a severe collapse (e.g. K=5 requested, only 2 clusters actually
    # populated) is impossible to misread as a normal K=2 result.
    if n_unique < cluster_labels.max() + 1:
        print(f"  [Metrics] WARNING: {int(cluster_labels.max() + 1 - n_unique)} "
              f"of the requested clusters are COMPLETELY EMPTY — only "
              f"{n_unique} clusters actually have any households.")
    print(f"  [Metrics] N={N}  clusters_populated={n_unique}  "
          f"sizes={dict(zip(unique_labels.tolist(), counts.tolist()))}")

    try:
        cap = min(5000, N)
        if N > cap:
            rng  = np.random.default_rng(42)
            idx  = rng.choice(N, cap, replace=False)
            Z_ss = Z[idx]; L_ss = cluster_labels[idx]
            if len(np.unique(L_ss)) < n_unique:
                Z_ss = Z; L_ss = cluster_labels
        else:
            Z_ss = Z; L_ss = cluster_labels
        sil = float(silhouette_score(Z_ss, L_ss, metric="euclidean"))
    except Exception as e:
        print(f"  [Metrics] Silhouette error: {e}")
        sil = np.nan

    try:
        dbi = float(davies_bouldin_score(Z, cluster_labels))
    except Exception as e:
        print(f"  [Metrics] DBI error: {e}")
        dbi = np.nan

    try:
        chi = float(calinski_harabasz_score(Z, cluster_labels))
    except Exception as e:
        print(f"  [Metrics] CHI error: {e}")
        chi = np.nan

    return {"silhouette": sil, "davies_bouldin": dbi,
            "calinski_harabasz": chi}


def evaluate_silhouette(encoder, X, cluster_labels, batch_size=256):
    return evaluate_metrics(encoder, X, cluster_labels, batch_size)["silhouette"]


# ===========================================================================
# Optional post-training refinement — "polish" pass
# ===========================================================================
#
# During DynAE joint training, the encoder is optimised for a MIX of
# reconstruction (L1) and compactness (L2) — the resulting embedding is
# not purely optimised for cluster separability. After training converges,
# freezing the encoder and re-fitting a fresh K-Means directly on the
# embeddings (with no reconstruction constraint pulling against it) often
# recovers extra separation "left on the table" by the joint objective.
#
# This is standard practice in DEC/IDEC-style literature (a final hard
# assignment pass on frozen embeddings) and is reported HERE TRANSPARENTLY
# alongside the original joint-training metrics — it never silently
# replaces them. Use with the `--polish` main.py flag.

def refine_labels_kmeans(encoder, X, K: int, batch_size: int = 256,
                         random_state: int = 42, n_init: int = 50) -> dict:
    """
    Freeze the trained encoder, re-embed X, and fit a fresh K-Means
    directly on the embeddings (no joint reconstruction/compactness loss).

    Returns
    -------
    dict with keys: labels, centroids, metrics (Sil/DBI/CHI on the
    refined assignment).
    """
    Z = encode_full(encoder, X, batch_size)
    if np.isnan(Z).any():
        print("  [Polish] WARNING: NaN in embeddings — skipping refinement.")
        return None

    km = KMeans(n_clusters=K, init="k-means++", n_init=n_init,
               random_state=random_state)
    refined_labels = km.fit_predict(Z)

    metrics = evaluate_metrics_from_Z(Z, refined_labels)
    return {
        "labels":    refined_labels,
        "centroids": km.cluster_centers_,
        "metrics":   metrics,
    }


def evaluate_metrics_from_Z(Z: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """Same metric computation as evaluate_metrics(), but takes a
    pre-computed embedding matrix Z directly (used by refine_labels_kmeans
    to avoid re-encoding X a second time)."""
    _NAN = {"silhouette": np.nan, "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan}
    N = Z.shape[0]
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    n_unique = len(unique_labels)

    if n_unique < 2 or counts.min() < 2 or N <= n_unique:
        return _NAN

    try:
        cap = min(5000, N)
        if N > cap:
            rng  = np.random.default_rng(42)
            idx  = rng.choice(N, cap, replace=False)
            Z_ss = Z[idx]; L_ss = cluster_labels[idx]
            if len(np.unique(L_ss)) < n_unique:
                Z_ss = Z; L_ss = cluster_labels
        else:
            Z_ss = Z; L_ss = cluster_labels
        sil = float(silhouette_score(Z_ss, L_ss, metric="euclidean"))
    except Exception:
        sil = np.nan
    try:
        dbi = float(davies_bouldin_score(Z, cluster_labels))
    except Exception:
        dbi = np.nan
    try:
        chi = float(calinski_harabasz_score(Z, cluster_labels))
    except Exception:
        chi = np.nan

    return {"silhouette": sil, "davies_bouldin": dbi,
            "calinski_harabasz": chi}


# ===========================================================================
# Clustering plots
# ===========================================================================

def plot_clustering_history(history: dict,
                             K:        int,
                             dataset:  str  = "dataset",
                             save_dir: str  = "results",
                             show:     bool = True):
    """Three plots per K: loss curves, metrics evolution, conflicted+tau_p."""
    import matplotlib.pyplot as plt, os

    out = f"{save_dir}/K{K}"
    os.makedirs(out, exist_ok=True)

    # 1. Loss curves
    L  = history.get("L",  [])
    L1 = history.get("L1", [])
    L2 = history.get("L2", [])
    if L:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Clustering Loss — {dataset.title()} K={K}",
                     fontsize=13, fontweight="bold")
        iters = range(1, len(L) + 1)
        for ax, vals, label, color in zip(
            axes, [L, L1, L2],
            ["Total L", "L1 (Recon/Centroid)", "L2 (Compactness raw)"],
            ["steelblue", "firebrick", "seagreen"]
        ):
            ax.plot(iters, vals, color=color, linewidth=1.2)
            ax.fill_between(iters, vals, alpha=0.10, color=color)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel("Inner Iteration", fontsize=9)
            ax.set_ylabel("Loss", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        p = f"{out}/cluster_loss_{dataset}_K{K}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {p}")
        if show: plt.show()
        plt.close()

    # 2. Metrics evolution
    sil = history.get("silhouette",        [])
    dbi = history.get("davies_bouldin",    [])
    chi = history.get("calinski_harabasz", [])
    if sil:
        outer_iters = list(range(1, len(sil) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Metrics Evolution — {dataset.title()} K={K}",
                     fontsize=13, fontweight="bold")
        for ax, vals, label, note, color in zip(
            axes, [sil, dbi, chi],
            ["Silhouette Score", "Davies-Bouldin Index",
             "Calinski-Harabasz Index"],
            ["higher better [-1,1]", "lower better [0,∞)",
             "higher better [0,∞)"],
            ["steelblue", "firebrick", "seagreen"]
        ):
            clean = [(x, v) for x, v in zip(outer_iters, vals)
                     if not np.isnan(v)]
            if clean:
                xs, ys = zip(*clean)
                ax.plot(xs, ys, color=color, linewidth=2.0,
                        marker="o", markersize=5)
                ax.fill_between(xs, ys, alpha=0.10, color=color)
            ax.set_title(f"{label}\n({note})", fontsize=10, fontweight="bold")
            ax.set_xlabel("Outer Iteration", fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        p = f"{out}/cluster_metrics_{dataset}_K{K}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {p}")
        if show: plt.show()
        plt.close()

    # 3. Conflicted + tau_p
    n_conf = history.get("n_conflicted", [])
    tau_p  = history.get("tau_p",        [])
    N_tot  = history.get("N_total",      None)
    if n_conf:
        outer_iters = list(range(1, len(n_conf) + 1))
        n_unconf    = [N_tot - c for c in n_conf] if N_tot else []
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Conflicted vs Unconflicted — {dataset.title()} K={K}",
            fontsize=13, fontweight="bold")
        ax = axes[0]
        ax.bar(outer_iters, n_conf,
               label="Conflicted (S̄)", color="firebrick", alpha=0.8)
        if n_unconf:
            ax.bar(outer_iters, n_unconf, bottom=n_conf,
                   label="Unconflicted (S)", color="seagreen", alpha=0.8)
        ax.set_title("Sample Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("Outer Iteration", fontsize=10)
        ax.set_ylabel("Number of Samples", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax = axes[1]
        ax.plot(outer_iters, tau_p, color="darkorange",
                linewidth=2.0, marker="o", markersize=5)
        ax.fill_between(outer_iters, tau_p, alpha=0.15, color="darkorange")
        ax.axhline(y=0.01, color="red", linestyle="--", linewidth=1.2,
                   label="Convergence threshold (1%)")
        ax.set_title("Training Progress τp", fontsize=11, fontweight="bold")
        ax.set_xlabel("Outer Iteration", fontsize=10)
        ax.set_ylabel("τp = |S̄| / N",   fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_ylim(0, max(tau_p) * 1.15 if tau_p else 1)
        plt.tight_layout()
        p = f"{out}/cluster_conflicted_{dataset}_K{K}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {p}")
        if show: plt.show()
        plt.close()

    # 4. Feature Drift (only if track_feature_drift=True was used)
    fd = history.get("FD", [])
    if fd:
        iters = range(1, len(fd) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(iters, fd, color="#7B1FA2", linewidth=1.0, alpha=0.85)
        ax.fill_between(iters, fd, alpha=0.08, color="#7B1FA2")
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1.0,
                   label="Orthogonal (no relationship)")
        ax.axhline(y=1.0, color="seagreen", linestyle=":", linewidth=1.0,
                   alpha=0.6, label="Perfect agreement")
        ax.axhline(y=-1.0, color="firebrick", linestyle=":", linewidth=1.0,
                   alpha=0.6, label="Perfect conflict")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            f"Feature Drift — {dataset.title()} K={K}\n"
            "cos(grad L1, grad L2) — do reconstruction and compactness "
            "objectives agree?",
            fontsize=12, fontweight="bold")
        ax.set_xlabel("Inner Iteration", fontsize=10)
        ax.set_ylabel("FD (cosine similarity)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        p = f"{out}/cluster_feature_drift_{dataset}_K{K}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {p}")
        if show: plt.show()
        plt.close()

    # 5. Inner-iteration Silhouette (only if track_inner_silhouette=True was used)
    inner_sil      = history.get("inner_sil", [])
    inner_sil_iter = history.get("inner_sil_iter", [])
    if inner_sil and inner_sil_iter:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(inner_sil_iter, inner_sil, "o-", color="#1976D2",
                linewidth=1.3, markersize=3, alpha=0.85)
        ax.fill_between(inner_sil_iter, inner_sil, alpha=0.08, color="#1976D2")

        # Mark outer-iteration boundaries — each is where centroids/
        # thresholds were last updated before this stretch of training
        outer_bounds = history.get("outer_boundaries", [])
        for i, b in enumerate(outer_bounds):
            if b > 0:  # skip the very first boundary at 0 (nothing before it)
                ax.axvline(x=b, color="gray", linestyle=":", linewidth=0.9,
                          alpha=0.6,
                          label="Outer iter boundary" if i == 1 else None)

        ax.set_title(
            f"Silhouette During Inner Iterations — {dataset.title()} K={K}\n"
            "(sampled periodically WITHIN each outer step's training block, "
            "against that step's fixed centroids)",
            fontsize=12, fontweight="bold")
        ax.set_xlabel("Global Inner Iteration (accumulates across all outer steps)",
                      fontsize=10)
        ax.set_ylabel("Silhouette Score (sample)", fontsize=10)
        if outer_bounds:
            ax.legend(fontsize=8, loc="best")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        p = f"{out}/cluster_inner_silhouette_{dataset}_K{K}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {p}")
        if show: plt.show()
        plt.close()

    # 6. [NEW] Sil + DBI + CHI during inner iterations (from track_best_
    #    checkpoint's monitoring — this data was being COLLECTED but never
    #    plotted until now; only Silhouette (via a separate mechanism,
    #    track_inner_silhouette) had a figure. This closes that gap.
    monitor_iter = history.get("monitor_iter", [])
    monitor_sil  = history.get("monitor_sil", [])
    monitor_dbi  = history.get("monitor_dbi", [])
    monitor_chi  = history.get("monitor_chi", [])

    if monitor_iter and monitor_sil:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        outer_bounds = history.get("outer_boundaries", [])

        panels = [
            (axes[0], monitor_sil, "Silhouette", "#1976D2", "higher is better"),
            (axes[1], monitor_dbi, "Davies-Bouldin", "#E53935", "lower is better"),
            (axes[2], monitor_chi, "Calinski-Harabasz", "#43A047", "higher is better"),
        ]
        for ax, values, label, color, note in panels:
            # Replace NaN with None so matplotlib skips gaps cleanly
            values_plot = [v if not (isinstance(v, float) and np.isnan(v))
                          else None for v in values]
            ax.plot(monitor_iter, values_plot, "o-", color=color,
                   linewidth=1.3, markersize=3, alpha=0.85)
            for i, b in enumerate(outer_bounds):
                if b > 0:
                    ax.axvline(x=b, color="gray", linestyle=":",
                              linewidth=0.8, alpha=0.5)
            ax.set_title(f"{label}\n({note})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Global Inner Iteration", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)

        # Mark the best checkpoint's iteration across all 3 panels, if one
        # was actually saved (track_best_checkpoint=True and a valid
        # balance-guard-passing best was found)
        best_iter = history.get("best_checkpoint_iter")
        if best_iter is not None:
            for ax in axes:
                ax.axvline(x=best_iter, color="black", linestyle="-",
                         linewidth=1.5, alpha=0.7)
            axes[0].text(best_iter, max([v for v in monitor_sil
                                        if not np.isnan(v)], default=0),
                       "  best\n  checkpoint", fontsize=8, va="top")

        fig.suptitle(
            f"Sil / DBI / CHI During Inner Iterations — "
            f"{dataset.title()} K={K}\n"
            "(sampled periodically WITHIN each outer step's training block, "
            "on the fixed monitoring subset; black line = restored best checkpoint)",
            fontsize=12, fontweight="bold")
        plt.tight_layout()
        p = f"{out}/cluster_inner_metrics_{dataset}_K{K}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {p}")
        if show: plt.show()
        plt.close()
