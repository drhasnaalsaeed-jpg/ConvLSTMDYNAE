"""
load_profiling.py
=================
Step 3 & 4 of Algorithm 2 — Load Profile Generation and Visualisation
(Section 5.4.3, Fig. 7, Fig. 8, Fig. 9)

Covers:
  • Building the load-profile matrix T  (a × K means, Algorithm 2 lines 17-22)
  • Daily resampling                    (Algorithm 2 lines 25-27)
  • Silhouette vs K curve (elbow/knee)  (Fig. 10, 11)
  • Silhouette score evolution plot     (Fig. 12)
  • Comparison charts                   (Fig. 13, 14)
  • Exporting profiles to CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import silhouette_score
from typing import Optional


# ---------------------------------------------------------------------------
# Optimal K selection — Elbow method + Knee method (cross-validated)
# ---------------------------------------------------------------------------
#
# Two independent, standard techniques are applied and cross-checked:
#
#   ELBOW METHOD  (distance-to-chord / "kneedle" geometry)
#     Applied to the raw Silhouette curve — the paper's primary metric.
#     Draws a straight chord from the first to the last K, and picks the
#     K whose point sits furthest above that chord. This is the classic
#     elbow criterion: the point of maximum deviation from linear decay.
#
#   KNEE METHOD  (discrete second-derivative / curvature)
#     Applied to the composite score (normalised Sil + 1/DBI + CHI).
#     Picks the K with the sharpest change in slope (max curvature),
#     i.e. where increasing K stops paying off — the formal "knee".
#
#   Both are reported side-by-side. When they agree, confidence is high.
#   When they disagree, the composite knee is used as the tie-breaker
#   because it accounts for all three metrics rather than Silhouette alone.

def _normalise(vals, higher_better=True):
    arr = np.array(vals, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.ones_like(arr) * 0.5
    normed = (arr - mn) / (mx - mn)
    return normed if higher_better else 1.0 - normed


def elbow_method_chord(ks: list, values: np.ndarray) -> tuple:
    """
    Classic elbow / kneedle: distance of each point from the straight
    chord connecting the first and last point. Returns (elbow_k, distances).
    """
    n = len(ks)
    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)
    slope     = (y[-1] - y[0]) / max(n - 1, 1)
    line_y    = y[0] + slope * x
    distances = y - line_y
    idx       = int(np.argmax(distances))
    return ks[idx], distances


def knee_method_curvature(ks: list, values: np.ndarray) -> tuple:
    """
    Discrete second-derivative knee detection. The knee is the K at which
    the curve bends most sharply (maximum |second derivative|), found via
    a central-difference approximation. Falls back to the chord method
    for very short curves (<3 points) where curvature is undefined.
    """
    n = len(ks)
    y = np.asarray(values, dtype=float)
    if n < 3:
        return elbow_method_chord(ks, values)

    second_deriv = np.zeros(n)
    for i in range(1, n - 1):
        second_deriv[i] = y[i - 1] - 2 * y[i] + y[i + 1]

    # Search ONLY interior points — endpoints have no second derivative.
    # BUG FIX: the previous version set endpoints to -np.inf intending to
    # exclude them, then took np.abs() before argmax. abs(-inf) == inf,
    # which ALWAYS wins an abs-magnitude argmax regardless of the real
    # interior curvature — silently forcing the knee to K[0] every time.
    # Restricting the search to the interior index range avoids the
    # sentinel entirely rather than trying to out-clever it with abs().
    interior_idx = np.arange(1, n - 1)
    best_interior = int(np.argmax(np.abs(second_deriv[interior_idx])))
    idx = int(interior_idx[best_interior])

    return ks[idx], second_deriv


def select_optimal_k(all_sil: dict, all_dbi: dict, all_chi: dict,
                     min_sil_quality:    float = 0.5,
                     min_cluster_frac:   dict  = None,
                     min_cluster_fraction: float = 0.03) -> dict:
    """
    Cross-validated optimal-K selection.

    The FINAL decision (`optimal_k`) is the K with the best composite
    score among "sane" candidates — this is deliberately more robust than
    picking via curvature/elbow detection, because with only a handful of
    K values, a degenerate partition at some intermediate K (e.g. one
    dominant cluster + a few near-empty fragments) can inflate raw
    Silhouette and make the composite curve sharply non-monotonic. Pure
    curvature-based "knee" detection then locks onto the sharpest LOCAL
    bend (often the crash point right after the best K), not the actual
    best value. Direct argmax over composite avoids that failure mode.

    Elbow (on raw Silhouette) and Knee (on composite curvature) are still
    computed and returned as cross-validation diagnostics — useful for
    sanity-checking and for the diagnostic plot — but they no longer
    override the final pick.

    Parameters
    ----------
    all_sil, all_dbi, all_chi : dict {K: value} for each metric
    min_sil_quality    : K values with Silhouette below this are excluded
                        (treated as collapsed).
    min_cluster_frac    : optional dict {K: smallest_cluster_size / N}.
                        When provided, any K whose smallest cluster falls
                        below `min_cluster_fraction` is EXCLUDED, since a
                        cluster with only a handful of members (e.g. 2-30
                        households out of thousands) is a degenerate
                        fragment, not a meaningful segment — and such
                        fragments are exactly what inflates Silhouette
                        while destroying DBI/CHI (see module docstring
                        example). Pass this whenever per-K label arrays
                        are available; skipping it risks selecting a
                        degenerate K purely because Silhouette was fooled.
    min_cluster_fraction : threshold for min_cluster_frac filtering (3%
                        default — tune based on how small a "real" segment
                        is plausible for your data).

    Returns
    -------
    dict with keys:
      optimal_k        : final chosen K — argmax(composite) among sane candidates
      elbow_k          : K chosen by the elbow method (Silhouette only) — diagnostic
      knee_k           : K chosen by the knee method (composite curvature) — diagnostic
      methods_agree    : bool — does elbow_k or knee_k match optimal_k?
      hq_ks            : list of high-quality K values considered
      composite        : np.ndarray of composite scores over hq_ks
      sil_n, dbi_n, chi_n : normalised per-metric arrays over hq_ks
      elbow_distances  : chord distances (Silhouette-based elbow)
      knee_curvature   : second-derivative curve (composite-based knee)
      excluded_degenerate : list of K values dropped by min_cluster_frac
    """
    ks = sorted(all_sil.keys())
    hq_ks = [k for k in ks
             if not np.isnan(all_sil[k]) and all_sil[k] >= min_sil_quality]

    excluded_degenerate = []
    if min_cluster_frac is not None:
        kept = []
        for k in hq_ks:
            frac = min_cluster_frac.get(k, 1.0)   # unknown → assume fine
            if frac < min_cluster_fraction:
                excluded_degenerate.append(k)
            else:
                kept.append(k)
        hq_ks = kept

    if len(hq_ks) == 0:
        return {"optimal_k": ks[0], "elbow_k": ks[0], "knee_k": ks[0],
                "methods_agree": True, "hq_ks": [], "composite": np.array([]),
                "sil_n": np.array([]), "dbi_n": np.array([]), "chi_n": np.array([]),
                "elbow_distances": np.array([]), "knee_curvature": np.array([]),
                "excluded_degenerate": excluded_degenerate,
                "fallback": "no_high_quality_k"}

    if len(hq_ks) == 1:
        k0 = hq_ks[0]
        return {"optimal_k": k0, "elbow_k": k0, "knee_k": k0,
                "methods_agree": True, "hq_ks": hq_ks, "composite": np.array([1.0]),
                "sil_n": np.array([1.0]), "dbi_n": np.array([1.0]), "chi_n": np.array([1.0]),
                "elbow_distances": np.array([0.0]), "knee_curvature": np.array([0.0]),
                "excluded_degenerate": excluded_degenerate,
                "fallback": "single_high_quality_k"}

    sil_vals = [all_sil[k] for k in hq_ks]
    dbi_vals = [all_dbi[k] for k in hq_ks]
    chi_vals = [all_chi[k] for k in hq_ks]

    sil_n = _normalise(sil_vals, higher_better=True)
    dbi_n = _normalise(dbi_vals, higher_better=False)   # lower DBI = better
    chi_n = _normalise(chi_vals, higher_better=True)
    composite = (sil_n + dbi_n + chi_n) / 3.0

    # Elbow method — raw Silhouette curve (paper's primary metric) — diagnostic
    elbow_k, elbow_dist = elbow_method_chord(hq_ks, np.array(sil_vals))

    # Knee method — composite score curvature — diagnostic
    knee_k, knee_curv = knee_method_curvature(hq_ks, composite)

    # FINAL decision: direct best composite score (robust to non-monotonic
    # curves), NOT the curvature-based knee.
    optimal_k = hq_ks[int(np.argmax(composite))]

    methods_agree = (elbow_k == optimal_k) or (knee_k == optimal_k)

    return {
        "optimal_k": optimal_k, "elbow_k": elbow_k, "knee_k": knee_k,
        "methods_agree": methods_agree, "hq_ks": hq_ks, "composite": composite,
        "sil_n": sil_n, "dbi_n": dbi_n, "chi_n": chi_n,
        "elbow_distances": elbow_dist, "knee_curvature": knee_curv,
        "excluded_degenerate": excluded_degenerate,
        "fallback": None,
    }


def print_k_selection_report(result: dict, dataset: str = ""):
    """Human-readable console report for select_optimal_k()'s output."""
    hq_ks = result["hq_ks"]

    excluded = result.get("excluded_degenerate", [])
    if excluded:
        print(f"\n[Select K] Excluded as degenerate (smallest cluster below "
              f"the balance threshold): {excluded}  [{dataset.upper()}]")
        print(f"  These K values likely have one dominant cluster plus tiny "
              f"outlier fragments — inflates Silhouette while DBI/CHI reveal "
              f"the true (poor) partition quality.")

    if not hq_ks:
        print(f"\n[Select K] No high-quality K values — "
              f"using K={result['optimal_k']} as fallback.  [{dataset.upper()}]")
        return

    print(f"\n[Select K] Candidates considered: {hq_ks}  [{dataset.upper()}]")
    if len(hq_ks) > 1:
        print(f"  {'K':>4}  {'Sil_n':>7}  {'DBI_n':>7}  {'CHI_n':>7}  "
              f"{'Composite':>10}")
        for i, k in enumerate(hq_ks):
            marks = ""
            if k == result["optimal_k"]: marks += " [SELECTED]"
            if k == result["elbow_k"]:   marks += " [elbow]"
            if k == result["knee_k"]:    marks += " [knee]"
            print(f"  {k:>4}  {result['sil_n'][i]:>7.4f}  "
                  f"{result['dbi_n'][i]:>7.4f}  {result['chi_n'][i]:>7.4f}  "
                  f"{result['composite'][i]:>10.4f}{marks}")

    print(f"\n  Final decision: BEST COMPOSITE SCORE → K={result['optimal_k']}")
    print(f"  (Elbow/Knee shown above are cross-validation diagnostics only "
          f"— they do not override the composite-score decision, since "
          f"curvature-based detection can lock onto a non-monotonic bend "
          f"rather than the actual best value.)")
    print(f"    Elbow method (Silhouette chord-distance) → K={result['elbow_k']}")
    print(f"    Knee  method (composite curvature)        → K={result['knee_k']}")
    if result["methods_agree"]:
        print(f"  Elbow or Knee agrees with the composite pick — "
              f"additional confidence in K={result['optimal_k']}")
    else:
        print(f"  Note: neither Elbow nor Knee matches the composite pick. "
              f"This can happen on non-monotonic curves (see excluded "
              f"K values above, if any) — the composite score remains the "
              f"most reliable signal since it weighs all three metrics "
              f"directly rather than inferring from curve shape.")


def plot_elbow_knee_analysis(result: dict,
                             title:     str = "Optimal K Selection",
                             save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Two-panel diagnostic figure:
      Left  — Silhouette curve with the ELBOW point marked (chord-distance,
              diagnostic only)
      Right — Composite-score curve with BOTH the KNEE point (curvature,
              diagnostic) and the actual SELECTED optimal_k (best composite
              score — the real decision) marked
    """
    hq_ks = result["hq_ks"]
    if len(hq_ks) < 2:
        print("[Plot] Skipping elbow/knee plot — fewer than 2 high-quality K values.")
        return None

    optimal_k = result["optimal_k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Elbow (Silhouette + chord line) — diagnostic only ────────
    x = np.arange(len(hq_ks))
    y = result["sil_n"]
    slope  = (y[-1] - y[0]) / max(len(hq_ks) - 1, 1)
    line_y = y[0] + slope * x

    ax1.plot(hq_ks, y, "o-", color="#1976D2", linewidth=2, label="Silhouette (norm.)")
    ax1.plot(hq_ks, line_y, "--", color="gray", linewidth=1.3, label="Chord line")
    ax1.axvline(result["elbow_k"], color="#E53935", linestyle=":", linewidth=2)
    ax1.scatter([result["elbow_k"]], [y[hq_ks.index(result["elbow_k"])]],
               color="#E53935", s=140, zorder=5,
               label=f"Elbow K={result['elbow_k']} (diagnostic)")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Normalised Silhouette")
    ax1.set_title("Elbow Method — Silhouette chord-distance\n(diagnostic only)")
    ax1.set_xticks(hq_ks)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # ── Right: Composite score — actual decision + knee diagnostic ─────
    comp = result["composite"]
    ax2.plot(hq_ks, comp, "o-", color="#43A047", linewidth=2,
             label="Composite score (Sil+1/DBI+CHI)")

    if result["knee_k"] != optimal_k:
        ax2.axvline(result["knee_k"], color="#FB8C00", linestyle=":", linewidth=1.5)
        ax2.scatter([result["knee_k"]], [comp[hq_ks.index(result["knee_k"])]],
                   color="#FB8C00", s=90, zorder=4, marker="^",
                   label=f"Knee K={result['knee_k']} (diagnostic only)")

    ax2.axvline(optimal_k, color="#E53935", linestyle="-", linewidth=2)
    ax2.scatter([optimal_k], [comp[hq_ks.index(optimal_k)]],
               color="#E53935", s=160, zorder=5,
               label=f"SELECTED K={optimal_k} (best composite score)")
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Composite score")
    ax2.set_title("Composite Score — actual selection\n(direct best-score, robust to non-monotonic curves)")
    ax2.set_xticks(hq_ks)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3)

    excluded = result.get("excluded_degenerate", [])
    excl_txt = f" | excluded as degenerate: {excluded}" if excluded else ""
    agree_txt = ("elbow/knee agree with selection"
                if result["methods_agree"] else
                "elbow/knee diverge from selection (non-monotonic curve)")
    fig.suptitle(f"{title}\nFinal choice: K={optimal_k}  ({agree_txt}){excl_txt}",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Build load-profile matrix T  (Algorithm 2, lines 16-22)
# ---------------------------------------------------------------------------

def build_load_profiles(wide: pd.DataFrame,
                        cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Compute the load profile for each cluster by averaging all household
    consumption time series belonging to that cluster.

    Parameters
    ----------
    wide           : (T_timestamps × N_households) normalised DataFrame
    cluster_labels : (N_households,) integer cluster assignment

    Returns
    -------
    T_matrix : DataFrame of shape (T_timestamps, K)
               Each column is the mean load curve for cluster k.
    """
    households = wide.columns.tolist()
    assert len(households) == len(cluster_labels), (
        f"Households ({len(households)}) ≠ labels ({len(cluster_labels)})"
    )

    K = int(cluster_labels.max()) + 1
    profiles = {}

    for k in range(K):
        mask = cluster_labels == k
        houses_k = [h for h, m in zip(households, mask) if m]
        if len(houses_k) == 0:
            # Empty cluster — fill with zeros
            profiles[f"Cluster_{k}"] = pd.Series(0.0, index=wide.index)
        else:
            profiles[f"Cluster_{k}"] = wide[houses_k].mean(axis=1)

    T_matrix = pd.DataFrame(profiles, index=wide.index)
    print(f"[Profile] Load profile matrix shape: {T_matrix.shape}  "
          f"(timestamps × K={K})")
    return T_matrix


# ---------------------------------------------------------------------------
# Daily resampling  (Algorithm 2, line 26)
# ---------------------------------------------------------------------------

def resample_daily(T_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Resample half-hourly load profiles to daily mean consumption.
    """
    daily = T_matrix.resample("D").mean()
    print(f"[Profile] Resampled to daily: {daily.shape}")
    return daily


# ---------------------------------------------------------------------------
# Plot load profiles  (Fig. 8 & 9)
# ---------------------------------------------------------------------------

def plot_load_profiles(T_matrix:   pd.DataFrame,
                       title:       str  = "Load Profiles",
                       ylabel:      str  = "Normalised Consumption",
                       figsize:     tuple = (12, 4),
                       save_path:   Optional[str] = None):
    """
    Plot one time-series per cluster  (Fig. 8 / 9 in the paper).
    """
    fig, ax = plt.subplots(figsize=figsize)

    for col in T_matrix.columns:
        ax.plot(T_matrix.index, T_matrix[col], label=col, linewidth=1.2)

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Customer Cluster", fontsize=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# All three metrics vs K  — three-panel plot  (replaces old Sil-only plot)
# ---------------------------------------------------------------------------

def plot_metrics_vs_k(metrics_by_k: dict,
                      title:        str  = "Clustering Metrics vs K",
                      save_path:    Optional[str] = None,
                      optimal_k:    Optional[int] = None):
    """
    Three-panel figure showing all three validity indices vs number of clusters.
    Marks the optimal K (selected by elbow on composite score) on all three panels.

    Parameters
    ----------
    metrics_by_k : dict  { method_name: { k: {"silhouette": v,
                                               "davies_bouldin": v,
                                               "calinski_harabasz": v} } }
    optimal_k    : int   The K selected as optimal — marked with a vertical
                         dashed line and annotation on all three panels.
    """
    metric_cfg = [
        ("silhouette",        "Silhouette Score",        "higher is better", "steelblue"),
        ("davies_bouldin",    "Davies-Bouldin Index",    "lower is better",  "firebrick"),
        ("calinski_harabasz", "Calinski-Harabasz Index", "higher is better", "seagreen"),
    ]

    # 4-panel figure: 3 individual metrics + composite elbow
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    markers = ["o-", "s--", "^:", "D-.", "v-"]
    all_ks = sorted(set(k for k_dict in metrics_by_k.values() for k in k_dict.keys()))

    # ── Panels 1-3: individual metrics ───────────────────────────────────
    for ax, (key, ylabel, note, color) in zip(axes[:3], metric_cfg):
        for (method, k_dict), marker in zip(metrics_by_k.items(), markers):
            ks   = sorted(k_dict.keys())
            vals = [k_dict[k][key] if isinstance(k_dict[k], dict)
                    else k_dict[k] for k in ks]
            # Replace NaN with None so matplotlib skips them
            vals_plot = [v if not np.isnan(v) else None for v in vals]
            ax.plot(ks, vals_plot, marker, label=method,
                    color=color if len(metrics_by_k) == 1 else None,
                    linewidth=1.8, markersize=6)

        if optimal_k is not None:
            ax.axvline(x=optimal_k, color="darkorange", linestyle="--",
                       linewidth=2.0, alpha=0.9, label=f"Optimal K={optimal_k}")
            ylim = ax.get_ylim()
            ax.annotate(
                f"K={optimal_k}\n(optimal)",
                xy=(optimal_k, ylim[1]),
                xytext=(optimal_k + 0.3, ylim[1]),
                fontsize=8, color="darkorange", fontweight="bold", va="top",
            )

        ax.set_xlabel("Number of Clusters (K)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel}\n({note})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_ks)

    # ── Panel 4: composite elbow curve ───────────────────────────────────
    ax4 = axes[3]

    # Recompute composite for all methods (use first method)
    method_name = list(metrics_by_k.keys())[0]
    k_dict      = metrics_by_k[method_name]

    def _safe_norm(vals, higher_better=True):
        arr = np.array(vals, dtype=float)
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx == mn:
            return np.ones_like(arr) * 0.5
        normed = (arr - mn) / (mx - mn)
        return normed if higher_better else 1.0 - normed

    valid_ks = [k for k in all_ks
                if not np.isnan(k_dict[k].get("silhouette", np.nan))
                and k_dict[k].get("silhouette", 0) >= 0.5]

    if len(valid_ks) >= 2:
        sil_v = [k_dict[k]["silhouette"]        for k in valid_ks]
        dbi_v = [k_dict[k]["davies_bouldin"]     for k in valid_ks]
        chi_v = [k_dict[k]["calinski_harabasz"]  for k in valid_ks]

        sil_n = _safe_norm(sil_v, True)
        dbi_n = _safe_norm(dbi_v, False)
        chi_n = _safe_norm(chi_v, True)
        comp  = (sil_n + dbi_n + chi_n) / 3.0

        # Elbow chord line
        n_pts   = len(valid_ks)
        x_arr   = np.arange(n_pts, dtype=float)
        slope   = (comp[-1] - comp[0]) / max(n_pts - 1, 1)
        line_y  = comp[0] + slope * x_arr
        dists   = comp - line_y

        ax4.plot(valid_ks, comp,   "D-",  color="#7B1FA2",
                 linewidth=2.0, markersize=7, label="Composite score")
        ax4.plot(valid_ks, line_y, "--",  color="gray",
                 linewidth=1.2, alpha=0.7, label="Chord line")

        # Mark elbow point
        elbow_idx = int(np.argmax(dists))
        elbow_k   = valid_ks[elbow_idx]
        ax4.scatter([elbow_k], [comp[elbow_idx]],
                    color="darkorange", s=120, zorder=5,
                    label=f"Elbow = K={elbow_k}")
        ax4.axvline(x=elbow_k, color="darkorange", linestyle="--",
                    linewidth=2.0, alpha=0.9)

        # Annotate composite scores
        for i, k in enumerate(valid_ks):
            ax4.annotate(f"{comp[i]:.3f}", (k, comp[i]),
                         textcoords="offset points", xytext=(0, 8),
                         fontsize=7, ha="center", color="#7B1FA2")

    ax4.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax4.set_ylabel("Composite Score (0–1)", fontsize=10)
    ax4.set_title("Elbow Method\n(Composite: Sil + 1/DBI + CHI normalised)",
                  fontsize=10, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(all_ks)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()


# Keep the old name as an alias for backward compatibility
def plot_silhouette_vs_k(sil_scores: dict,
                         title:      str = "Silhouette Score vs K",
                         save_path:  Optional[str] = None):
    """
    Backward-compatible wrapper. Accepts the old format:
        { method_name: [(k, sil), ...] }
    and converts it for plot_metrics_vs_k.
    """
    converted = {}
    for method, scores in sil_scores.items():
        converted[method] = {k: {"silhouette": v,
                                 "davies_bouldin": np.nan,
                                 "calinski_harabasz": np.nan}
                             for k, v in scores}
    plot_metrics_vs_k(converted, title=title, save_path=save_path)


# ---------------------------------------------------------------------------
# Three-metric evolution across clustering iterations
# ---------------------------------------------------------------------------

def plot_metrics_evolution(history:   dict,
                           title:     str = "Clustering Metrics Evolution",
                           save_path: Optional[str] = None):
    """
    Three-panel plot showing how each metric evolves across outer iterations.

    Parameters
    ----------
    history : dict from cluster() containing keys:
              'silhouette', 'davies_bouldin', 'calinski_harabasz'
    """
    metric_cfg = [
        ("silhouette",        "Silhouette Score",
         "higher is better",  "steelblue"),
        ("davies_bouldin",    "Davies-Bouldin Index",
         "lower is better",   "firebrick"),
        ("calinski_harabasz", "Calinski-Harabasz Index",
         "higher is better",  "seagreen"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for ax, (key, ylabel, note, color) in zip(axes, metric_cfg):
        vals = history.get(key, [])
        if not vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            iters = range(1, len(vals) + 1)
            ax.plot(iters, vals, color=color, linewidth=2.0, marker="o",
                    markersize=5)
            ax.fill_between(iters, vals, alpha=0.10, color=color)

        ax.set_xlabel("Outer Iteration", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel}\n({note})", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()


# Keep old name as alias
def plot_silhouette_evolution(sil_history: list,
                              title:       str = "Silhouette Score Evolution",
                              save_path:   Optional[str] = None):
    """Backward-compatible wrapper."""
    plot_metrics_evolution({"silhouette": sil_history},
                           title=title, save_path=save_path)


# ---------------------------------------------------------------------------
# Three-metric summary table — printed and saved to CSV
# ---------------------------------------------------------------------------

def print_metrics_table(all_results: dict, dataset: str = ""):
    """
    Print a formatted table of all three metrics across K values,
    and return a DataFrame.

    Parameters
    ----------
    all_results : { k: {"sil": v, "dbi": v, "chi": v} }
    """
    rows = []
    for k in sorted(all_results.keys()):
        r = all_results[k]
        rows.append({
            "K":                      k,
            "Silhouette Score":       round(r.get("sil",  np.nan), 4),
            "Davies-Bouldin Index":   round(r.get("dbi",  np.nan), 4),
            "Calinski-Harabasz Index":round(r.get("chi",  np.nan), 2),
        })

    df = pd.DataFrame(rows).set_index("K")

    header = f"\n{'='*62}"
    if dataset:
        header += f"\n  Clustering Metrics — {dataset.title()} Dataset"
    header += f"\n{'='*62}"
    print(header)
    print(df.to_string())
    print(f"{'='*62}")
    print("  Silhouette Score        : higher is better  (range [-1,  1])")
    print("  Davies-Bouldin Index    : lower  is better  (range [ 0,  ∞))")
    print("  Calinski-Harabasz Index : higher is better  (range [ 0,  ∞))")
    print(f"{'='*62}\n")

    return df


# ---------------------------------------------------------------------------
# Comparison chart vs baselines  (Fig. 13 & 14)
# ---------------------------------------------------------------------------

def plot_comparison(results_dict: dict,
                    title:        str = "Comparison with Previous Works",
                    save_path:    Optional[str] = None):
    """
    Parameters
    ----------
    results_dict : {method_name: {k: sil_score}}
                   e.g. {
                     "ConvLSTM-DynAE": {2: 0.990, 3: 0.996, ...},
                     "DynAE":          {2: 0.740, 3: 0.616, ...},
                   }
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    markers = ["o-", "s-", "^-", "D-", "v-"]
    for (method, scores), marker in zip(results_dict.items(), markers):
        ks   = sorted(scores.keys())
        vals = [scores[k] for k in ks]
        ax.plot(ks, vals, marker, label=method, linewidth=1.5)

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score (Sil)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Grid search for optimal K  (Algorithm 2, lines 7-12)
# ---------------------------------------------------------------------------

def grid_search_k(X:          np.ndarray,
                  encoder:    "tf.keras.Model",
                  cluster_fn: callable,
                  k_range:    range = range(2, 11),
                  **cluster_kwargs) -> dict:
    """
    Run clustering for each K in k_range, compute Silhouette Index.

    Returns
    -------
    results : {k: {"labels": ..., "sil": ...}}
    """
    results = {}

    for k in k_range:
        print(f"\n{'='*50}")
        print(f"Grid search: K = {k}")
        print("="*50)

        labels, history, centroids = cluster_fn(
            X, k=k, **cluster_kwargs
        )

        # Silhouette in latent space
        z_all = []
        bs = cluster_kwargs.get("batch_size", 256)
        for i in range(0, X.shape[0], bs):
            z_all.append(encoder(X[i:i + bs], training=False).numpy())
        Z = np.vstack(z_all)

        n_unique = len(np.unique(labels))
        if n_unique >= 2:
            sil = float(silhouette_score(Z, labels, metric="euclidean",
                                         sample_size=min(5000, Z.shape[0])))
        else:
            sil = -1.0

        results[k] = {
            "labels":    labels,
            "sil":       sil,
            "history":   history,
            "centroids": centroids,
        }
        print(f"K={k} → Silhouette = {sil:.4f}")

    return results


# ---------------------------------------------------------------------------
# Export load profiles to CSV
# ---------------------------------------------------------------------------

def export_profiles(T_matrix:  pd.DataFrame,
                    filepath:  str = "load_profiles.csv"):
    """Save the load-profile matrix to CSV."""
    T_matrix.to_csv(filepath)
    print(f"[Export] Load profiles saved to {filepath}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy wide dataframe
    np.random.seed(42)
    n_hh, n_ts = 50, 48 * 7
    idx = pd.date_range("2011-11-28", periods=n_ts, freq="30min")
    wide = pd.DataFrame(np.random.rand(n_ts, n_hh),
                        index=idx,
                        columns=[f"H{i}" for i in range(n_hh)])

    labels = np.random.randint(0, 3, n_hh)
    T_mat  = build_load_profiles(wide, labels)
    daily  = resample_daily(T_mat)

    print(f"Load profile matrix: {T_mat.shape}")
    print(f"Daily resampled:     {daily.shape}")
    print("load_profiling.py smoke test passed.")
