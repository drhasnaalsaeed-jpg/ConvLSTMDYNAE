"""
main.py
=======
Full ConvLSTM-DynAE Load Profiling Pipeline (Algorithm 2, Section 5.4)

Memory-safe. Results are saved SEPARATELY per dataset under:
    results/<dataset>/pretrain/        ← pretraining weights + plots
    results/<dataset>/cluster/K<k>/    ← per-K clustering weights + plots
    results/<dataset>/profiles/        ← load profile CSVs + plots
    results/<dataset>/consumption/     ← 6 consumption pattern plots

Usage
-----
  # London — free Colab (12 GB RAM)
  python main.py --dataset london \
                 --data_path data/london/LCL-FullData.csv \
                 --output_dir results/london \
                 --cache_dir  data/cache \
                 --max_houses 1500

  # London — Colab Pro (25 GB RAM, all households)
  python main.py --dataset london \
                 --data_path data/london/LCL-FullData.csv \
                 --output_dir results/london \
                 --cache_dir  data/cache

  # Irish dataset
  python main.py --dataset irish \
                 --data_path data/irish/ \
                 --output_dir results/irish \
                 --cache_dir  data/cache

  # Skip pretraining (load saved weights)
  python main.py --dataset london ... --skip_pretrain
"""

import argparse
import time
import numpy as np
import pandas as pd
import gc
import psutil
from pathlib import Path
import tensorflow as tf

from data_preprocessing import (preprocess_london, preprocess_irish,
                                 check_ram, clear_cache)
from model              import build_autoencoder, build_critic
from pretrain           import pretrain, plot_pretrain_history
from clustering         import (cluster, evaluate_metrics,
                                 plot_clustering_history,
                                 refine_labels_kmeans)
from load_profiling     import (build_load_profiles, resample_daily,
                                 plot_load_profiles,
                                 plot_metrics_vs_k,
                                 plot_metrics_evolution,
                                 print_metrics_table,
                                 export_profiles,
                                 select_optimal_k,
                                 print_k_selection_report,
                                 plot_elbow_knee_analysis)
from consumption_plots  import plot_all_consumption_patterns


# ---------------------------------------------------------------------------
# Defaults  (Section 6.2.2)
# ---------------------------------------------------------------------------
DEFAULTS = {
    # ── Architecture (Section 6.2.2, Fig. 3) ──────────────────────────────
    "n_steps":       7,        # days/week → input (N, 7, 1, 48, 1)
    "n_length":     48,        # half-hours/day
    "n_features":    1,
    "latent_dim":   10,        # paper Section 6.2.2
    "dropout":       0.2,

    # ── Pretraining (Section 6.2.2) ───────────────────────────────────────
    "pretrain_lr":   1e-4,     # Adam lr=0.0001
    "pretrain_iter": 1000,     # 1000 iterations (paper value)
    "lambda_acai":   0.1,      # λ in Eq. 17 — best value found empirically
                               # gives K=3 Sil=0.9143 for London (stable result)
    "use_lambda_ramp": False,  # OFF: ramping λ 0->0.1 was tested empirically
                               # and reduced Z_var/Silhouette vs constant λ

    # ── Clustering (Section 6.2.2 / Algorithm 1) ──────────────────────────
    "cluster_lr":    1e-3,     # SGD lr=0.001 (paper)
    "momentum":      0.9,      # paper
    "max_iter":      1500,     # Updated from paper's 1000 after a clean,
                               # isolated A/B/C test (same seed=3407,
                               # same skip_pretrain weights, Irish K=3):
                               #   1000 -> Sil=0.8449 DBI=0.2031 CHI=64080
                               #   1500 -> Sil=0.8645 DBI=0.1792 CHI=74582  (best on all 3)
                               #   2000 -> Sil=0.8144 DBI=0.2539 CHI=47715
                               # 1500 is a genuine sweet spot, not "more is
                               # better" — likely reflects real tightening
                               # up to a point, then overfitting to a
                               # soon-to-be-revised pseudo-label snapshot
                               # beyond it. NOTE: only validated at K=3 on
                               # Irish with this one seed — may not be the
                               # exact optimum for other K/dataset/seed
                               # combinations, but is a better-evidenced
                               # starting point than the paper's 1000.
    "max_outer":     15,       # increased from 10 — more outer iters for convergence
    "l2_weight":     0.05,     # reduced from 0.1 — gentler L2 allows better K=3 separation
    "batch_size":   256,
    "total_thresh":  0.01,     # convergence τp < 1% (paper)
    "kappa_drop":    0.3,      # κ drop rate=0.3 (paper)
    "kappa_min_fraction": 0.5,   # kappa_min = kappa_init * this — reverted
                                  # after A/B test showed 0.5 beats 0.02
                                  # on Sil/DBI/CHI (see clustering.py)
    "kmeans_n_init": 50,         # K-Means restarts (init + every regen).
                                  # Untested against alternatives so far —
                                  # 50 is the REFINEMENTS-spec value, not
                                  # yet A/B validated like max_iter/kappa_min.
    "always_update_centroids": True,  # paper Fig.6: unconditional update
                                       # each outer iter. Set False to use
                                       # the nairouz/DynAE repo's event-
                                       # triggered variant instead.
    "k_min":         2,
    "k_max":        10,
    "min_cluster_fraction": 0.03,  # K's with a cluster smaller than this
                                   # fraction of N are excluded as degenerate
                                   # before elbow/knee/composite K-selection

    # [NEW] Pretraining stability improvements (pretrain.py §1-§3)
    "warm_up_iters": 300,      # reconstruction-only iters before ACAI starts
    "es_patience":   150,      # EarlyStopping patience (iters)
    "es_min_delta":  1e-5,     # min val loss improvement to reset patience
    "lr_patience":   80,       # LR plateau patience (iters)
    "lr_factor":     0.5,      # LR multiplied by this on plateau
    "lr_min":        1e-6,     # minimum learning rate
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def free_memory(*_):
    import ctypes
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except: pass

def set_global_seed(seed: int = 42):
    """
    Fix all random seeds for full reproducibility.
    Sources of randomness in this pipeline:
      1. Python random module
      2. NumPy random
      3. TensorFlow / Keras weight initialisation
      4. TensorFlow ops (dropout, shuffle)
      5. K-Means initialisation (handled per call via random_state=seed)
    """
    import random, os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[Seed] Global random seed set to {seed} "
          f"(Python / NumPy / TensorFlow)")

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try: tf.config.experimental.set_memory_growth(g, True)
            except: pass
        print(f"[GPU] Memory growth enabled for {len(gpus)} GPU(s).")
    else:
        print("[GPU] No GPU — running on CPU.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    cfg     = DEFAULTS.copy()
    dataset = args.dataset

    # ── Compute-time tracking ───────────────────────────────────────────────
    # Every major phase is timed independently and reported in a summary
    # table + saved CSV at the end (mirrors nairouz/DynAE's time()-based
    # phase timers, e.g. "Clustering time: %ds").
    t_pipeline_start = time.time()
    timings = {}   # phase_name -> seconds

    # Fix all random seeds first — before any model or data initialisation
    set_global_seed(args.seed)

    # Apply any CLI overrides to DEFAULTS
    if args.l2_weight  is not None: cfg["l2_weight"]  = args.l2_weight
    if args.max_iter   is not None: cfg["max_iter"]   = args.max_iter
    if args.pretrain_iter is not None: cfg["pretrain_iter"] = args.pretrain_iter
    if args.max_outer  is not None: cfg["max_outer"]  = args.max_outer
    if args.kappa_drop is not None: cfg["kappa_drop"] = args.kappa_drop
    if args.kappa_min_fraction is not None:
        cfg["kappa_min_fraction"] = args.kappa_min_fraction
    if args.kmeans_n_init is not None:
        cfg["kmeans_n_init"] = args.kmeans_n_init
    if args.k_min      is not None: cfg["k_min"]      = args.k_min
    if args.k_max      is not None: cfg["k_max"]      = args.k_max
    if args.min_cluster_fraction is not None:
        cfg["min_cluster_fraction"] = args.min_cluster_fraction
    if args.event_triggered_centroids:
        cfg["always_update_centroids"] = False

    # ── Separate output folders per dataset ───────────────────────────────
    base_dir     = Path(args.output_dir)
    pretrain_dir = base_dir / "pretrain"
    cluster_dir  = base_dir / "cluster"
    profile_dir  = base_dir / "profiles"
    consume_dir  = base_dir / "consumption"
    cache_dir    = args.cache_dir if args.cache_dir else str(base_dir / "cache")

    for d in [pretrain_dir, cluster_dir, profile_dir, consume_dir,
              Path(cache_dir)]:
        d.mkdir(parents=True, exist_ok=True)

    setup_gpu()

    # ── RAM check + auto max_houses ────────────────────────────────────────
    print("\n" + "="*60)
    print(f"SYSTEM CHECK — {dataset.upper()} DATASET")
    print("="*60)
    check_ram()
    avail_gb = psutil.virtual_memory().available / 1e9
    if args.max_houses is None and avail_gb < 18 and dataset == "london":
        auto = 1500 if avail_gb < 10 else 2500
        print(f"\n  WARNING: {avail_gb:.1f} GB RAM. "
              f"Auto-setting max_houses={auto} to avoid crash.")
        args.max_houses = auto

    # ================================================================== #
    # STEP 1 — Preprocessing                                              #
    # ================================================================== #
    print("\n" + "="*60)
    print(f"STEP 1 — DATA PREPROCESSING  [{dataset.upper()}]")
    print("="*60)
    t_preprocess_start = time.time()

    if dataset == "london":
        X, wide = preprocess_london(
            filepath          = args.data_path,
            n_steps           = cfg["n_steps"],
            n_length          = cfg["n_length"],
            max_houses        = args.max_houses,
            chunk_size        = args.chunk_size,
            cache_dir         = cache_dir,
            force_reprocess   = args.force_reprocess,
            missing_threshold = args.missing_threshold,
            informations_households_path = args.london_informations_path,
        )
    else:
        X, wide = preprocess_irish(
            data_dir          = args.data_path,
            n_steps           = cfg["n_steps"],
            n_length          = cfg["n_length"],
            cache_dir         = cache_dir,
            force_reprocess   = args.force_reprocess,
            missing_threshold = args.missing_threshold,
            allocation_path   = args.irish_allocation_path,
        )

    timings["preprocessing"] = time.time() - t_preprocess_start
    print(f"[Timing] Preprocessing took {timings['preprocessing']:.1f}s "
          f"({timings['preprocessing']/60:.2f} min)")

    print(f"\nInput tensor shape: {X.shape}")
    check_ram()

    # ── Build models ───────────────────────────────────────────────────────
    encoder, decoder, _ = build_autoencoder(
        n_steps=cfg["n_steps"], n_length=cfg["n_length"],
        n_features=cfg["n_features"], latent_dim=cfg["latent_dim"],
        dropout_rate=cfg["dropout"],
    )
    critic = build_critic(
        n_steps=cfg["n_steps"], n_length=cfg["n_length"],
        n_features=cfg["n_features"],
    )
    _ = encoder(X[:2]); _ = decoder(encoder(X[:2]))

    # ================================================================== #
    # STEP 2a — Phase I: Pretraining                                      #
    # ================================================================== #
    print("\n" + "="*60)
    print(f"STEP 2a — PHASE I: PRETRAINING  [{dataset.upper()}]")
    print(f"         Weights → {pretrain_dir}/")
    print("="*60)
    t_pretrain_start = time.time()

    if not args.skip_pretrain:
        check_ram()
        pretrain_history = pretrain(
            X                = X,
            encoder          = encoder,
            decoder          = decoder,
            critic           = critic,
            max_iter         = cfg["pretrain_iter"],
            batch_size       = cfg["batch_size"],
            lr               = cfg["pretrain_lr"],
            lam              = cfg["lambda_acai"],
            mode             = args.pretrain_mode,
            # [NEW] training stability improvements
            # use_lambda_ramp=False: empirically confirmed (seed=42, London,
            # K=3) that ramping halves average adversarial pressure and
            # lowers Z_var (8.76->6.48) and Silhouette (0.8756->0.7792).
            use_lambda_ramp  = cfg.get("use_lambda_ramp", False),
            warm_up_iters    = cfg.get("warm_up_iters", 300),
            patience         = cfg.get("es_patience",   150),
            min_delta        = cfg.get("es_min_delta",  1e-5),
            restore_best     = True,
            lr_patience      = cfg.get("lr_patience",   80),
            lr_factor        = cfg.get("lr_factor",     0.5),
            lr_min           = cfg.get("lr_min",        1e-6),
            monitor_interval = 100,
            save_pca         = args.save_pca,
            save_tsne        = args.save_tsne,
            vis_dir          = str(pretrain_dir / "vis"),
            dataset          = dataset,
            save_weights     = str(pretrain_dir),
            verbose          = 100,
        )
        # ── Save pretraining plots ────────────────────────────────────────
        plot_pretrain_history(
            pretrain_history,
            dataset  = dataset,
            save_dir = str(pretrain_dir),
            show     = False,
        )
        # Save history CSV
        # ae_loss, val_loss, critic_loss, lr are logged every iteration.
        # z_var, silhouette, active_ratio, latent_norm are logged at
        # monitor_interval — pad with last value to align lengths.
        n = len(pretrain_history.get("ae_loss", []))
        history_csv = {
            "ae_loss":     pretrain_history.get("ae_loss",     [np.nan]*n),
            "val_loss":    pretrain_history.get("val_loss",    [np.nan]*n),
            "critic_loss": pretrain_history.get("critic_loss", [np.nan]*n),
            "lr":          pretrain_history.get("lr",          [np.nan]*n),
        }
        def _pad_to(lst, length):
            """Repeat last value to fill up to `length` entries."""
            if not lst:
                return [np.nan] * length
            import math
            step = max(1, math.ceil(length / len(lst)))
            out  = []
            for v in lst:
                out.extend([v] * step)
            return out[:length]

        for key in ["z_var", "silhouette", "davies_bouldin",
                    "calinski_harabasz", "active_ratio", "latent_norm"]:
            history_csv[key] = _pad_to(
                pretrain_history.get(key, []), n)

        pd.DataFrame(history_csv).to_csv(
            str(pretrain_dir / f"pretrain_history_{dataset}.csv"), index=False
        )
        print(f"[Pretrain] History CSV saved.")
        check_ram()
    else:
        print(f"[Skip pretrain] Loading weights from {pretrain_dir}/")
        encoder.load_weights(str(pretrain_dir / "encoder.weights.h5"))
        decoder.load_weights(str(pretrain_dir / "decoder.weights.h5"))
        pretrain_history = {}

    timings["pretraining"] = time.time() - t_pretrain_start
    print(f"[Timing] Pretraining phase took {timings['pretraining']:.1f}s "
          f"({timings['pretraining']/60:.2f} min)")

    del critic; free_memory()

    if args.pretrain_only:
        z_var_hist = pretrain_history.get("z_var", [])
        final_zvar = z_var_hist[-1] if z_var_hist else float("nan")
        timings["total_pipeline"] = time.time() - t_pipeline_start
        print("\n" + "="*60)
        print(f"[Pretrain-only] Stopping before clustering (--pretrain_only).")
        print(f"[Pretrain-only] SEED={args.seed}  FINAL_ZVAR={final_zvar:.4f}")
        print(f"[Timing] Preprocessing : {timings['preprocessing']:.1f}s")
        print(f"[Timing] Pretraining   : {timings['pretraining']:.1f}s")
        print(f"[Timing] Total         : {timings['total_pipeline']:.1f}s")
        print("="*60)
        return

    # ================================================================== #
    # STEP 2b — Grid search over K                                        #
    # ================================================================== #
    print("\n" + "="*60)
    print(f"STEP 2b — GRID SEARCH K={cfg['k_min']}..{cfg['k_max']}  "
          f"[{dataset.upper()}]")
    print(f"         Cluster weights → {cluster_dir}/K<k>/")
    print("="*60)

    sil_by_k   = {}
    all_results = {}
    t_clustering_start = time.time()
    per_k_times = {}

    for k in range(cfg["k_min"], cfg["k_max"] + 1):
        print(f"\n{'─'*40}")
        print(f"  K = {k}  [{dataset.upper()}]")
        print(f"{'─'*40}")
        check_ram()
        t_k_start = time.time()

        enc_k, dec_k, _ = build_autoencoder(
            cfg["n_steps"], cfg["n_length"], cfg["n_features"],
            cfg["latent_dim"], cfg["dropout"],
        )
        _ = enc_k(X[:2]); _ = dec_k(enc_k(X[:2]))
        enc_k.load_weights(str(pretrain_dir / "encoder.weights.h5"))
        dec_k.load_weights(str(pretrain_dir / "decoder.weights.h5"))

        k_dir = str(cluster_dir / f"K{k}")

        cached_labels_path = Path(k_dir) / "cluster_labels.npy"
        cached_enc_path    = Path(k_dir) / "encoder_clustered.weights.h5"
        cached_dec_path    = Path(k_dir) / "decoder_clustered.weights.h5"
        cached_cents_path  = Path(k_dir) / "centroids.npy"

        if (getattr(args, "skip_clustering", False)
                and cached_labels_path.exists()
                and cached_enc_path.exists()):
            print(f"[Skip clustering] Reloading cached K={k} results from "
                  f"{k_dir}/ (no re-training — labels/weights/centroids "
                  f"already computed with the current clustering.py logic)")
            enc_k.load_weights(str(cached_enc_path))
            if cached_dec_path.exists():
                dec_k.load_weights(str(cached_dec_path))
            labels_k = np.load(str(cached_labels_path))
            cents_k  = (np.load(str(cached_cents_path))
                       if cached_cents_path.exists() else None)
            hist_k   = {}   # no training happened — nothing to plot/export
        else:
            if getattr(args, "skip_clustering", False):
                print(f"[Skip clustering] No cached results found for K={k} "
                      f"at {k_dir}/ — running cluster() normally.")
            labels_k, hist_k, cents_k = cluster(
                X               = X,
                encoder         = enc_k,
                decoder         = dec_k,
                K               = k,
                max_iter        = cfg["max_iter"],
                max_outer       = cfg["max_outer"],
                batch_size      = cfg["batch_size"],
                lr              = cfg["cluster_lr"],
                momentum        = cfg["momentum"],
                kappa_drop_rate = cfg["kappa_drop"],
                kappa_min_fraction = cfg.get("kappa_min_fraction", 0.5),
                kmeans_n_init   = cfg.get("kmeans_n_init", 50),
                always_update_centroids = cfg.get("always_update_centroids", True),
                total_thresh    = cfg["total_thresh"],
                l2_weight       = cfg["l2_weight"],
                save_path       = k_dir,
                track_feature_drift = args.track_feature_drift,
                track_inner_silhouette = args.track_inner_silhouette,
                seed                = args.seed,
                track_best_checkpoint = args.track_best_checkpoint,
                checkpoint_monitor_interval = args.checkpoint_monitor_interval,
                checkpoint_min_cluster_fraction = args.checkpoint_min_cluster_fraction,
                checkpoint_composite_weights = (
                    args.checkpoint_sil_weight,
                    (1 - args.checkpoint_sil_weight - args.checkpoint_balance_weight) / 2,
                    (1 - args.checkpoint_sil_weight - args.checkpoint_balance_weight) / 2,
                    args.checkpoint_balance_weight),
                checkpoint_min_valid_for_composite = args.checkpoint_min_valid_for_composite,
                clustering_early_stopping = args.clustering_early_stopping,
                clustering_patience = args.clustering_patience,
                verbose         = 20,
            )

        per_k_times[k] = time.time() - t_k_start
        print(f"[Timing] K={k} clustering took {per_k_times[k]:.1f}s "
              f"({per_k_times[k]/60:.2f} min)")

        # ── Save clustering plots per K ───────────────────────────────────
        plot_clustering_history(
            hist_k,
            K        = k,
            dataset  = dataset,
            save_dir = str(cluster_dir),
            show     = False,
        )

        # Save clustering history CSV
        # Lists in history may have different lengths:
        #   L, L1, L2          — one entry per INNER iteration
        #   silhouette, dbi, chi, tau_p, n_conflicted — one per OUTER iteration
        # Save them in two separate CSVs to avoid the length mismatch error.
        inner_keys = ["L", "L1", "L2", "FD"]   # FD only populated when
                                               # --track_feature_drift is set
        outer_keys = ["silhouette", "davies_bouldin", "calinski_harabasz",
                      "tau_p", "n_conflicted"]

        k_csv_dir = cluster_dir / f"K{k}"
        k_csv_dir.mkdir(parents=True, exist_ok=True)

        # Inner iteration history (loss curves)
        inner_data = {key: hist_k[key] for key in inner_keys
                      if key in hist_k and hist_k[key]}
        if inner_data:
            pd.DataFrame(inner_data).to_csv(
                str(k_csv_dir / f"cluster_loss_history_{dataset}_K{k}.csv"),
                index=False
            )

        # Outer iteration history (metrics + convergence)
        outer_data = {key: hist_k[key] for key in outer_keys
                      if key in hist_k and hist_k[key]}
        if outer_data:
            # Pad shorter lists to the same length as the longest
            max_len = max(len(v) for v in outer_data.values())
            padded  = {k: v + [np.nan] * (max_len - len(v))
                       for k, v in outer_data.items()}
            pd.DataFrame(padded).to_csv(
                str(k_csv_dir / f"cluster_outer_history_{dataset}_K{k}.csv"),
                index=False
            )

        # ── Compute metrics ───────────────────────────────────────────────
        n_found = len(np.unique(labels_k))

        # Diagnose and handle bad clustering outcomes
        if n_found < 2:
            print(f"  [K={k}] WARNING: All samples collapsed into 1 cluster.")
            print("  Metrics (Sil/DBI/CHI) are undefined with only 1 cluster.")
            print("  Possible causes:")
            print("    1. Pretraining was insufficient — try more iterations")
            print("    2. Dataset has too little variation after imputation")
            print("    3. K is too large for the data structure")
            print(f"  Recording NaN for K={k} and continuing to next K...")
            m = {"silhouette": np.nan,
                 "davies_bouldin": np.nan,
                 "calinski_harabasz": np.nan}
        elif n_found < k:
            print(f"  [K={k}] WARNING: Requested {k} clusters but "
                  f"only {n_found} non-empty clusters found.")
            print(f"  Computing metrics on the {n_found} actual clusters.")
            m = evaluate_metrics(enc_k, X, labels_k, cfg["batch_size"])
        else:
            m = evaluate_metrics(enc_k, X, labels_k, cfg["batch_size"])

        sil_by_k[k] = m["silhouette"]
        polish_used = False
        m_joint = dict(m)   # keep the original joint-training metrics too

        # ── Optional post-training polish (--polish flag) ──────────────────
        # Freezes the trained encoder and re-fits a plain K-Means directly
        # on the embeddings, with no reconstruction/compactness constraint
        # pulling against separation. Reported transparently: BOTH the
        # original joint-training metrics and the polished metrics are
        # printed and saved; the polished labels/metrics are only adopted
        # for this K if they are genuinely better (higher Silhouette).
        if getattr(args, "polish", False) and n_found >= 2:
            polish = refine_labels_kmeans(enc_k, X, K=n_found,
                                          batch_size=cfg["batch_size"],
                                          random_state=args.seed)
            if polish is not None:
                pm = polish["metrics"]
                pm_s   = f"{pm['silhouette']:.4f}"        if not np.isnan(pm['silhouette'])        else "NaN"
                pdbi_s = f"{pm['davies_bouldin']:.4f}"    if not np.isnan(pm['davies_bouldin'])    else "NaN"
                pchi_s = f"{pm['calinski_harabasz']:.2f}" if not np.isnan(pm['calinski_harabasz'])  else "NaN"
                joint_sil_s = f"{m_joint['silhouette']:.4f}" if not np.isnan(m_joint['silhouette']) else "NaN"
                print(f"  [Polish] K-Means-on-frozen-embeddings metrics:")
                print(f"    Silhouette Score        = {pm_s}  (joint-training was {joint_sil_s})")
                print(f"    Davies-Bouldin Index    = {pdbi_s}")
                print(f"    Calinski-Harabasz Index = {pchi_s}")

                if (not np.isnan(pm["silhouette"]) and
                    (np.isnan(m["silhouette"]) or pm["silhouette"] > m["silhouette"])):
                    print(f"  [Polish] Polished Silhouette is higher — "
                          f"adopting refined labels for K={k}.")
                    labels_k    = polish["labels"]
                    cents_k     = polish["centroids"]
                    m           = pm
                    sil_by_k[k] = pm["silhouette"]
                    polish_used = True
                else:
                    print(f"  [Polish] Joint-training result already as good or "
                          f"better — keeping original labels for K={k}.")

        all_results[k] = {
            "labels":           labels_k,
            "sil":              m["silhouette"],
            "dbi":              m["davies_bouldin"],
            "chi":              m["calinski_harabasz"],
            "history":          hist_k,
            "centroids":        cents_k,
            "encoder":          enc_k,
            "n_clusters_found": n_found,
            "polish_used":      polish_used,
            "joint_sil":        m_joint["silhouette"],
            "time_sec":         per_k_times[k],
        }

        sil_s = f"{m['silhouette']:.4f}"   if not np.isnan(m['silhouette'])        else "NaN"
        dbi_s = f"{m['davies_bouldin']:.4f}" if not np.isnan(m['davies_bouldin'])  else "NaN"
        chi_s = f"{m['calinski_harabasz']:.2f}" if not np.isnan(m['calinski_harabasz']) else "NaN"
        print(f"\n  K={k} final metrics [{dataset.upper()}]"
              f"{'  [POLISHED]' if polish_used else ''}:")
        print(f"    Silhouette Score        = {sil_s}  (higher better)")
        print(f"    Davies-Bouldin Index    = {dbi_s}  (lower  better)")
        print(f"    Calinski-Harabasz Index = {chi_s}  (higher better)")

        del dec_k; free_memory()

    timings["clustering_total"]   = time.time() - t_clustering_start
    timings["clustering_per_k"]   = per_k_times
    avg_k_time = (timings["clustering_total"] / len(per_k_times)
                 if per_k_times else 0.0)
    print(f"\n[Timing] Grid search over K={cfg['k_min']}..{cfg['k_max']} "
          f"took {timings['clustering_total']:.1f}s "
          f"({timings['clustering_total']/60:.2f} min total, "
          f"{avg_k_time:.1f}s avg per K)")

    # ── Select optimal K using composite score (Sil + DBI + CHI) ──────────
    # Algorithm 2, Step 13 (Section 5.4.2) — enhanced with cross-validation:
    #
    #   FINAL decision: direct argmax of the composite score among "sane"
    #                 candidates. Robust to non-monotonic curves — see
    #                 select_optimal_k() docstring for why curvature-based
    #                 knee detection alone is NOT used as the decider.
    #   ELBOW method: chord-distance on raw Silhouette — reported as a
    #                 cross-validation diagnostic only.
    #   KNEE   method: discrete second-derivative curvature on the
    #                 composite score — reported as a diagnostic only.
    #
    # min_cluster_frac excludes degenerate K values (one dominant cluster +
    # tiny outlier fragments) BEFORE any of the above runs, since such
    # partitions inflate raw Silhouette while DBI/CHI correctly reveal them
    # as poor. Without this, a degenerate K can be mistaken for the best
    # answer purely because Silhouette was fooled by a few far-flung,
    # internally-tiny "clusters" of 2-30 households.

    ks   = sorted(sil_by_k.keys())
    sils = [sil_by_k[k] for k in ks]

    all_sil = {k: all_results[k]["sil"] for k in ks}
    all_dbi = {k: all_results[k]["dbi"] for k in ks}
    all_chi = {k: all_results[k]["chi"] for k in ks}

    min_cluster_frac = {}
    for k in ks:
        labels_k_arr = all_results[k]["labels"]
        if labels_k_arr is not None and len(labels_k_arr) > 0:
            unique_labels, counts = np.unique(labels_k_arr, return_counts=True)
            n_populated = len(unique_labels)
            if n_populated < k:
                # [FIX] Some requested clusters ended up COMPLETELY EMPTY
                # (e.g. K=5 requested, only 2 clusters actually got any
                # households). The old check only looked at the smallest
                # POPULATED cluster's size, which this can slip past
                # entirely if the populated ones happen to be reasonably
                # sized — exactly what happened on a real London K=5 run.
                # Force this to fail the balance guard outright, since
                # "fewer distinct groups than requested" is a real form
                # of degeneracy the size-only check can't see.
                print(f"[K-selection] K={k}: only {n_populated}/{k} "
                      f"clusters are populated ({k - n_populated} "
                      f"completely empty) — forcing degenerate exclusion.")
                min_cluster_frac[k] = 0.0
            else:
                min_cluster_frac[k] = float(counts.min()) / len(labels_k_arr)

    k_selection = select_optimal_k(
        all_sil, all_dbi, all_chi,
        min_sil_quality      = 0.5,
        min_cluster_frac     = min_cluster_frac,
        min_cluster_fraction = cfg.get("min_cluster_fraction", 0.03),
    )
    print_k_selection_report(k_selection, dataset=dataset)

    optimal_k = k_selection["optimal_k"]
    hq_ks     = k_selection["hq_ks"]
    composite = k_selection["composite"]

    plot_elbow_knee_analysis(
        k_selection,
        title     = f"Optimal K Selection — {dataset.title()} Dataset",
        save_path = str(base_dir / f"elbow_knee_analysis_{dataset}.png"),
    )

    # ── Select the WORST K (lowest composite score) for contrast ─────────
    # Uses the same normalised Sil+DBI+CHI composite as optimal-K selection,
    # but takes the minimum instead of the elbow point. Falls back to the
    # worst raw Silhouette among all tested K if fewer than 2 high-quality
    # K values were found (mirrors the optimal_k fallback logic above).
    if len(hq_ks) >= 2:
        worst_idx = int(np.argmin(composite))
        worst_k   = hq_ks[worst_idx]
        print(f"[Select K] Worst K = {worst_k} "
              f"(lowest composite score = {composite[worst_idx]:.4f})")
    else:
        valid_all = [(k, all_sil[k]) for k in ks if not np.isnan(all_sil[k])]
        if valid_all:
            worst_k = min(valid_all, key=lambda t: t[1])[0]
        else:
            worst_k = ks[-1]
        print(f"[Select K] Worst K = {worst_k} (fallback: lowest raw Silhouette)")

    # Guard: don't let best and worst collapse to the same K when only a
    # couple of high-quality candidates exist — fall back to the lowest raw
    # Silhouette among the *other* K values instead.
    if worst_k == optimal_k:
        other = [(k, all_sil[k]) for k in ks
                 if k != optimal_k and not np.isnan(all_sil[k])]
        if other:
            worst_k = min(other, key=lambda t: t[1])[0]
            print(f"[Select K] Worst K collided with Optimal K — "
                  f"using next-worst distinct K={worst_k} instead.")
        else:
            print(f"[Select K] Only one usable K ({optimal_k}) — "
                  f"best and worst will be the same.")

    valid = [(k, all_sil[k]) for k in ks if not np.isnan(all_sil[k])]
    valid_str = ", ".join([f"K={k}:{s:.4f}" for k, s in valid])
    print(f"\n[Select K] All valid K: {valid_str}")
    print(f"[Select K] Optimal (best) K = {optimal_k}  [{dataset.upper()}]")
    print(f"[Select K] Worst K          = {worst_k}  [{dataset.upper()}]")
    print(f"[Select K] Method: Elbow (Silhouette chord) + Knee (composite curvature), "
          f"cross-validated  →  elbow_K={k_selection['elbow_k']}  knee_K={k_selection['knee_k']}")

    # Single-line machine-parseable summary — used by automated seed sweeps.
    # Field order (OPTIMAL_K immediately followed by SIL/DBI/CHI) is kept
    # backward-compatible with the Cell 3c sweep regex; new elbow/knee
    # diagnostic fields are appended at the end.
    _best = all_results[optimal_k]
    print(f"[RUN_SUMMARY] SEED={args.seed} DATASET={dataset} "
          f"OPTIMAL_K={optimal_k} "
          f"SIL={_best['sil']:.6f} "
          f"DBI={_best['dbi']:.6f} "
          f"CHI={_best['chi']:.6f} "
          f"ELBOW_K={k_selection['elbow_k']} "
          f"KNEE_K={k_selection['knee_k']} "
          f"METHODS_AGREE={k_selection['methods_agree']}")

    print_metrics_table(all_results, dataset=dataset)

    # Save metrics CSV
    rows = [{
        "K":                       k,
        "Silhouette Score":        round(all_results[k]["sil"], 4)
                                   if not np.isnan(all_results[k]["sil"]) else "NaN",
        "Davies-Bouldin Index":    round(all_results[k]["dbi"], 4)
                                   if not np.isnan(all_results[k]["dbi"]) else "NaN",
        "Calinski-Harabasz Index": round(all_results[k]["chi"], 2)
                                   if not np.isnan(all_results[k]["chi"]) else "NaN",
        "Optimal":                 k == optimal_k,
        "Worst":                   k == worst_k,
    } for k in ks]
    pd.DataFrame(rows).set_index("K").to_csv(
        str(base_dir / f"metrics_all_k_{dataset}.csv")
    )
    print(f"[Metrics] Saved → {base_dir}/metrics_all_k_{dataset}.csv")

    # Three-metric vs K plot
    metrics_for_plot = {
        "ConvLSTM-DynAE": {
            k: {"silhouette":        all_results[k]["sil"],
                "davies_bouldin":    all_results[k]["dbi"],
                "calinski_harabasz": all_results[k]["chi"]}
            for k in ks
        }
    }
    plot_metrics_vs_k(
        metrics_for_plot,
        title      = f"Clustering Metrics vs K — {dataset.title()} Dataset",
        save_path  = str(base_dir / f"metrics_vs_k_{dataset}.png"),
        optimal_k  = optimal_k,
    )

    # ================================================================== #
    # STEP 3-5 — Load profiles + metrics evolution + consumption plots,   #
    #            generated once for the BEST K and once for the WORST K  #
    # ================================================================== #
    n_hh = X.shape[0]
    if wide.shape[1] > n_hh:
        wide = wide.iloc[:, :n_hh]

    def _generate_k_outputs(k, label):
        """label is 'best' or 'worst' — controls sub-folder naming so the
        two runs never overwrite each other."""
        result_k = all_results[k]

        print("\n" + "="*60)
        print(f"STEP 3 — LOAD PROFILE GENERATION  "
              f"[{dataset.upper()} | {label.upper()} K={k}]")
        print("="*60)

        k_profile_dir = profile_dir / f"{label}_K{k}"
        k_profile_dir.mkdir(parents=True, exist_ok=True)

        T_matrix = build_load_profiles(wide, result_k["labels"])
        daily    = resample_daily(T_matrix)

        export_profiles(T_matrix, str(k_profile_dir / f"load_profiles_halfhourly_{dataset}.csv"))
        export_profiles(daily,    str(k_profile_dir / f"load_profiles_daily_{dataset}.csv"))

        plot_load_profiles(
            daily,
            title     = f"Load Profiles — {dataset.title()} {label.title()} K={k}",
            save_path = str(k_profile_dir / f"load_profiles_{dataset}_{label}_K{k}.png"),
        )

        plot_metrics_evolution(
            result_k["history"],
            title     = f"Metrics Evolution — {dataset.title()} {label.title()} K={k}",
            save_path = str(base_dir / f"metrics_evolution_{dataset}_{label}_K{k}.png"),
        )

        print("\n" + "="*60)
        print(f"STEP 5 — CONSUMPTION PATTERN PLOTS  "
              f"[{dataset.upper()} | {label.upper()} K={k}]")
        print(f"         (daily / 24h-intraday / weekly / monthly / seasonal / comparison)")
        print("="*60)

        k_consume_dir = consume_dir / f"{label}_K{k}"
        plot_all_consumption_patterns(
            T_matrix   = T_matrix,
            output_dir = str(k_consume_dir),
            dataset    = f"{dataset}_{label}_K{k}",
        )

        return T_matrix

    t_profiles_start = time.time()

    print(f"\n{'#'*60}\n# BEST K = {optimal_k}\n{'#'*60}")
    T_matrix_best = _generate_k_outputs(optimal_k, "best")

    print(f"\n{'#'*60}\n# WORST K = {worst_k}\n{'#'*60}")
    T_matrix_worst = _generate_k_outputs(worst_k, "worst")

    timings["profile_generation"] = time.time() - t_profiles_start
    print(f"\n[Timing] Profile + consumption-plot generation "
          f"(best K={optimal_k} + worst K={worst_k}) took "
          f"{timings['profile_generation']:.1f}s "
          f"({timings['profile_generation']/60:.2f} min)")

    del wide; free_memory()

    # ================================================================== #
    # Summary                                                             #
    # ================================================================== #
    print("\n" + "="*60)
    print(f"ALL STEPS COMPLETE  [{dataset.upper()}]")
    print("="*60)
    check_ram()
    print(f"""
Output structure:
  {base_dir}/
  ├── metrics_all_k_{dataset}.csv          ← all 3 metrics for K=2..10 (Optimal + Worst flagged)
  ├── metrics_vs_k_{dataset}.png           ← three-panel K comparison + composite elbow
  ├── metrics_evolution_{dataset}_best_K{optimal_k}.png
  ├── metrics_evolution_{dataset}_worst_K{worst_k}.png
  ├── pretrain/
  │   ├── encoder.weights.h5
  │   ├── decoder.weights.h5
  │   ├── pretrain_loss_{dataset}.png      ← AE loss + Critic loss panels
  │   ├── pretrain_loss_combined_{dataset}.png
  │   └── pretrain_history_{dataset}.csv
  ├── cluster/
  │   └── K<k>/  (for each K=2..10)
  │       ├── encoder_clustered.weights.h5
  │       ├── decoder_clustered.weights.h5
  │       ├── centroids.npy
  │       ├── cluster_labels.npy
  │       ├── cluster_history_{dataset}_K<k>.csv
  │       ├── cluster_loss_{dataset}_K<k>.png      ← L, L1, L2 curves
  │       ├── cluster_metrics_{dataset}_K<k>.png   ← Sil/DBI/CHI evolution
  │       └── cluster_conflicted_{dataset}_K<k>.png ← conflicted/unconflicted
  ├── profiles/
  │   ├── best_K{optimal_k}/
  │   │   ├── load_profiles_halfhourly_{dataset}.csv
  │   │   ├── load_profiles_daily_{dataset}.csv
  │   │   └── load_profiles_{dataset}_best_K{optimal_k}.png
  │   └── worst_K{worst_k}/
  │       ├── load_profiles_halfhourly_{dataset}.csv
  │       ├── load_profiles_daily_{dataset}.csv
  │       └── load_profiles_{dataset}_worst_K{worst_k}.png
  └── consumption/
      ├── best_K{optimal_k}/
      │   ├── 01_daily_consumption_{dataset}_best_K{optimal_k}.png
      │   ├── 02_intraday_consumption_{dataset}_best_K{optimal_k}.png   ← 24-hour
      │   ├── 03_weekly_consumption_{dataset}_best_K{optimal_k}.png
      │   ├── 04_monthly_consumption_{dataset}_best_K{optimal_k}.png
      │   ├── 05_seasonal_consumption_{dataset}_best_K{optimal_k}.png
      │   └── 06_cluster_comparison_grid_{dataset}_best_K{optimal_k}.png
      └── worst_K{worst_k}/
          ├── 01_daily_consumption_{dataset}_worst_K{worst_k}.png
          ├── 02_intraday_consumption_{dataset}_worst_K{worst_k}.png   ← 24-hour
          ├── 03_weekly_consumption_{dataset}_worst_K{worst_k}.png
          ├── 04_monthly_consumption_{dataset}_worst_K{worst_k}.png
          ├── 05_seasonal_consumption_{dataset}_worst_K{worst_k}.png
          └── 06_cluster_comparison_grid_{dataset}_worst_K{worst_k}.png
""")

    # ================================================================== #
    # Compute-time summary                                                #
    # ================================================================== #
    timings["total_pipeline"] = time.time() - t_pipeline_start

    def _fmt_time(sec):
        m, s = divmod(sec, 60)
        h, m = divmod(int(m), 60)
        return f"{h}h {m:02d}m {s:04.1f}s" if h else f"{int(m)}m {s:04.1f}s"

    print("\n" + "="*60)
    print(f"COMPUTE TIME SUMMARY  [{dataset.upper()}]")
    print("="*60)
    print(f"  {'Phase':<28} {'Seconds':>10}   {'Human-readable':>16}")
    print("  " + "-"*58)
    print(f"  {'Preprocessing':<28} {timings['preprocessing']:>10.1f}   "
          f"{_fmt_time(timings['preprocessing']):>16}")
    print(f"  {'Pretraining':<28} {timings['pretraining']:>10.1f}   "
          f"{_fmt_time(timings['pretraining']):>16}")
    print(f"  {'Clustering (all K, total)':<28} "
          f"{timings['clustering_total']:>10.1f}   "
          f"{_fmt_time(timings['clustering_total']):>16}")
    for k, t in sorted(timings["clustering_per_k"].items()):
        print(f"    K={k:<25} {t:>10.1f}   {_fmt_time(t):>16}")
    print(f"  {'Profile + plot generation':<28} "
          f"{timings['profile_generation']:>10.1f}   "
          f"{_fmt_time(timings['profile_generation']):>16}")
    print("  " + "-"*58)
    print(f"  {'TOTAL PIPELINE':<28} {timings['total_pipeline']:>10.1f}   "
          f"{_fmt_time(timings['total_pipeline']):>16}")
    print("="*60)

    # Save as CSV — one row per phase, plus one row per K within clustering
    timing_rows = [
        {"phase": "preprocessing",             "seconds": timings["preprocessing"]},
        {"phase": "pretraining",                "seconds": timings["pretraining"]},
        {"phase": "clustering_total_all_k",     "seconds": timings["clustering_total"]},
        {"phase": "profile_generation",         "seconds": timings["profile_generation"]},
        {"phase": "total_pipeline",             "seconds": timings["total_pipeline"]},
    ]
    for k, t in sorted(timings["clustering_per_k"].items()):
        timing_rows.append({"phase": f"clustering_K{k}", "seconds": t})

    timing_df = pd.DataFrame(timing_rows)
    timing_csv_path = base_dir / f"timing_summary_{dataset}.csv"
    timing_df.to_csv(str(timing_csv_path), index=False)
    print(f"[Timing] Summary saved → {timing_csv_path}")

    print(f"[RUN_SUMMARY_TIMING] SEED={args.seed} DATASET={dataset} "
          f"TOTAL_SEC={timings['total_pipeline']:.2f} "
          f"PREPROCESS_SEC={timings['preprocessing']:.2f} "
          f"PRETRAIN_SEC={timings['pretraining']:.2f} "
          f"CLUSTER_TOTAL_SEC={timings['clustering_total']:.2f} "
          f"PROFILES_SEC={timings['profile_generation']:.2f}")

    return all_results, optimal_k, worst_k


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ConvLSTM-DynAE — Memory-Safe Load Profiling Pipeline"
    )
    p.add_argument("--dataset",    choices=["london", "irish"], required=True)
    p.add_argument("--data_path",  required=True)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--cache_dir",  default=None,
                   help="Directory for caching preprocessed tensors")
    p.add_argument("--skip_pretrain", action="store_true")
    p.add_argument("--skip_clustering", action="store_true", default=False,
                   help="Reload cached per-K clustering results (labels, "
                        "centroids, clustered weights) from cluster/K<k>/ "
                        "instead of re-running cluster(). Use this when "
                        "only the K-SELECTION logic changed (e.g. a "
                        "load_profiling.py update) and the underlying "
                        "per-K clustering itself is still valid — skips "
                        "the most expensive step in the whole pipeline. "
                        "Falls back to running cluster() normally for any "
                        "K where cached files are missing.")
    p.add_argument("--max_houses", type=int, default=None,
                   help="Max households (None=all, 1500=free Colab safe)")
    p.add_argument("--chunk_size", type=int, default=500_000,
                   help="CSV rows per chunk (reduce to 200000 if crashing)")
    p.add_argument("--pretrain_mode", default="acai",
                   choices=["acai", "recon"],
                   help="Pretraining mode: 'acai' (paper, adversarial) or "
                        "'recon' (simple MSE, stable fallback). "
                        "Default=recon (more stable, prevents latent collapse).")
    p.add_argument("--force_reprocess", action="store_true",
                   help="Delete existing cache and re-read raw data from scratch. "
                        "Use this whenever the cached household count looks wrong.")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed for full reproducibility "
                        "(Python / NumPy / TensorFlow / K-Means). Default=42.")
    p.add_argument("--polish", action="store_true", default=False,
                   help="After DynAE joint training converges for each K, "
                        "freeze the encoder and re-fit a plain K-Means on "
                        "the embeddings (no reconstruction constraint). "
                        "Adopts the refined labels only if Silhouette "
                        "genuinely improves; both results are always "
                        "printed for transparency.")
    p.add_argument("--l2_weight",   type=float, default=None,
                   help="Override l2_weight in DEFAULTS.")
    p.add_argument("--max_iter",    type=int,   default=None,
                   help="Override max_iter (inner iterations per outer "
                        "step, CLUSTERING phase) in DEFAULTS. Paper value "
                        "is 1000; validated best is 1500 (see README).")
    p.add_argument("--pretrain_iter", type=int, default=None,
                   help="[NEW] Override pretrain_iter (PRETRAINING phase "
                        "iterations, Phase I / ACAI) in DEFAULTS. Paper "
                        "value is 1000. Distinct from --max_iter, which "
                        "only affects the clustering phase. Protected by "
                        "EarlyStopping (patience=150) and best-checkpoint-"
                        "by-val-loss restoration, so raising this is "
                        "relatively low-risk — worst case it stops early "
                        "with no change; it should not make pretraining "
                        "worse.")
    p.add_argument("--max_outer",   type=int,   default=None,
                   help="Override max_outer in DEFAULTS.")
    p.add_argument("--kappa_drop",  type=float, default=None,
                   help="Override kappa_drop in DEFAULTS.")
    p.add_argument("--kappa_min_fraction", type=float, default=None,
                   help="Override kappa_min_fraction (kappa_min = kappa_init "
                        "* this). DEFAULT IS 0.5 — a clean A/B test (same "
                        "seed, same pretrained weights) showed 0.5 beats an "
                        "aggressive 0.02 floor on Sil/DBI/CHI simultaneously; "
                        "0.02 lets the confidence threshold decay too fast, "
                        "rushing households out of self-supervision too early.")
    p.add_argument("--kmeans_n_init", type=int, default=None,
                   help="Override the number of K-Means restarts (default "
                        "50) used at initial centroid construction and every "
                        "centroid regeneration during training. Higher = "
                        "more exhaustive search for the best K-Means "
                        "initialisation, at the cost of more compute per "
                        "regeneration event. Does NOT affect baselines.py, "
                        "which is deliberately left at sklearn defaults "
                        "(n_init=20) for a fair, untuned comparison point.")
    p.add_argument("--event_triggered_centroids", action="store_true", default=False,
                   help="Use the nairouz/DynAE repo's event-triggered centroid "
                        "update (only regenerate on stall) instead of the "
                        "paper's Fig.6 unconditional every-iteration update "
                        "(the default). The two sources genuinely disagree "
                        "here — this flag lets you A/B test both.")
    p.add_argument("--k_min", type=int, default=None,
                   help="Override k_min in DEFAULTS (grid search start).")
    p.add_argument("--k_max", type=int, default=None,
                   help="Override k_max in DEFAULTS (grid search end).")
    p.add_argument("--min_cluster_fraction", type=float, default=None,
                   help="Override min_cluster_fraction in DEFAULTS. K values "
                        "whose smallest cluster falls below this fraction of "
                        "N are excluded from K-selection as degenerate "
                        "(one dominant cluster + tiny fragments).")
    p.add_argument("--pretrain_only", action="store_true", default=False,
                   help="Run pretraining + latent diagnostics only, then exit "
                        "before clustering. Use for fast seed screening — "
                        "check Z_var without paying the full clustering cost.")
    # [NEW] pretrain.py improvement arguments
    p.add_argument("--save_pca",  action="store_true", default=False,
                   help="Save PCA latent plots every 500 pretrain iters.")
    p.add_argument("--save_tsne", action="store_true", default=False,
                   help="Save t-SNE latent plot at end of pretraining.")
    p.add_argument("--track_feature_drift", action="store_true", default=False,
                   help="Track Feature Drift (FD) during clustering — cosine "
                        "similarity between grad(L1) and grad(L2) per inner "
                        "iteration. Diagnostic only (nairouz/DynAE concept, "
                        "never wired up in the original repo). Adds a second "
                        "backward pass per inner iteration when enabled.")
    p.add_argument("--track_inner_silhouette", action="store_true", default=False,
                   help="Track Silhouette periodically DURING the inner "
                        "loop (not just once per outer iteration), on a "
                        "fixed 2000-household sample against the current "
                        "frozen centroids. Shows whether Sil improves "
                        "monotonically within an outer step's training "
                        "block, or fluctuates/degrades. Adds encode+"
                        "silhouette cost every 50 inner iterations.")
    p.add_argument("--track_best_checkpoint", dest="track_best_checkpoint",
                   action="store_true", default=True,
                   help="[REFINEMENTS §2] Every --checkpoint_monitor_interval "
                        "inner iterations, evaluate Sil/DBI/CHI/sizes on a "
                        "fixed monitoring subset and save the encoder+decoder "
                        "if this is the best-so-far AND passes the degenerate-"
                        "cluster balance guard. Restores the BEST checkpoint "
                        "(not the final training iteration) before saving "
                        "final results. On by default.")
    p.add_argument("--no_track_best_checkpoint", dest="track_best_checkpoint",
                   action="store_false",
                   help="Disable best-checkpoint tracking — use the final "
                        "training iteration's weights as-is (pre-refinement "
                        "behaviour).")
    p.add_argument("--checkpoint_monitor_interval", type=int, default=100,
                   help="[REFINEMENTS §2] How often (in inner iterations) "
                        "to evaluate the best-checkpoint monitoring metrics.")
    p.add_argument("--checkpoint_min_cluster_fraction", type=float, default=0.03,
                   help="[REFINEMENTS §15] A checkpoint is only eligible to "
                        "become the new 'best' if its smallest cluster is at "
                        "least this fraction of N — rejects degenerate/"
                        "imbalanced partitions from being saved as best "
                        "purely because of an inflated Silhouette.")
    p.add_argument("--checkpoint_sil_weight", type=float, default=0.4,
                   help="[NEW] Weight given to Silhouette in the best-"
                        "checkpoint composite score (DBI and CHI evenly "
                        "split whatever remains after --checkpoint_"
                        "balance_weight is subtracted). Default 0.4.")
    p.add_argument("--checkpoint_balance_weight", type=float, default=0.2,
                   help="[NEW] Weight given to cluster-size BALANCE "
                        "(normalised entropy — 1.0 = perfectly even sizes) "
                        "in the composite score. This is separate from "
                        "--checkpoint_min_cluster_fraction, which only "
                        "rejects a checkpoint below a hard floor (e.g. "
                        "smallest cluster < 3%) but does not otherwise "
                        "prefer more balanced partitions over less "
                        "balanced ones that still clear that floor. "
                        "Default 0.2 actively rewards balance, not just "
                        "tolerates it.")
    p.add_argument("--checkpoint_min_valid_for_composite", type=int, default=3,
                   help="[NEW] Fall back to pure Silhouette-only comparison "
                        "until at least this many balance-guard-passing "
                        "checkpoints have been seen, THEN switch to the "
                        "weighted composite score. Running min-max "
                        "normalisation is maximally volatile with very few "
                        "points — with only 2, whichever is worse on any "
                        "single metric gets normalised to exactly 0 for "
                        "that metric.")
    p.add_argument("--clustering_early_stopping", action="store_true", default=False,
                   help="[REFINEMENTS §3] Optional additional inner-loop "
                        "early stop based on monitoring-metric plateau. Off "
                        "by default — the outer loop's own tau_p convergence "
                        "check is the primary, already-validated stopping "
                        "mechanism; this adds a stricter inner-loop-level "
                        "stop on top of it.")
    p.add_argument("--clustering_patience", type=int, default=500,
                   help="[REFINEMENTS §3] Inner iterations without monitoring "
                        "improvement before --clustering_early_stopping "
                        "triggers.")
    p.add_argument("--missing_threshold", type=float, default=1.0,
                   help="Drop households exceeding this fraction of missing readings."
                        " Default=1.0 (keep ALL households, impute all missing). "
                        " This matches the paper which kept ~5567 London households. "
                        " Use 0.50 to drop households with >50%% missing readings.")
    p.add_argument("--irish_allocation_path", type=str, default=None,
                   help="[NEW] Path to the CER 'SME and Residential "
                        "allocations' file (.xlsx or .csv). If given, "
                        "restricts the Irish dataset to Code==1 "
                        "(Residential) meter IDs only, excluding SME/"
                        "commercial meters that the raw File1-6.txt files "
                        "otherwise mix in with no type label. Ignored for "
                        "London (the London dataset does not have this "
                        "residential/SME mixing issue). Only takes effect "
                        "on a fresh read — pass --force_reprocess if a "
                        "cache already exists from before filtering.")
    p.add_argument("--london_informations_path", type=str, default=None,
                   help="[NEW] Path to Kaggle's 'informations_households"
                        ".csv' — ONLY needed if --data_path points at the "
                        "Kaggle 'Smart meters in London' halfhourly_dataset "
                        "(concatenated block files) rather than the "
                        "original raw LCL-FullData.csv. The Kaggle block "
                        "files have no inline tariff (stdorToU) column, so "
                        "without this, the default stdorToU_filter='Std' "
                        "silently filters nothing and Std+ToU households "
                        "get mixed together with no warning. Not needed "
                        "if using the original LCL-FullData.csv, which has "
                        "the tariff column inline already.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
