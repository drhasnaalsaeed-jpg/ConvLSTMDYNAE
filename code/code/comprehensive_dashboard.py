"""
comprehensive_dashboard.py
===========================
Seven SEPARATE publication figures — each its own standalone PNG+PDF,
not combined into one dashboard — matching the requested reference
design but split so each can be individually referenced as its own
Figure in a paper:

  plot_daily_trend_iqr()          -> Fig_A_DailyTrend
  plot_typical_daily_profile()    -> Fig_B_TypicalDailyProfile (original scale)
  plot_typical_daily_profile_normalized() -> Fig_C_TypicalDailyProfileShape
  plot_datetime_heatmap()         -> Fig_D_DateTimeHeatmap
  plot_weekly_profile_iqr()       -> Fig_E_WeeklyProfile
  plot_monthly_profile_iqr()      -> Fig_F_MonthlyProfile
  plot_seasonal_profile_iqr()     -> Fig_G_SeasonalProfile

------------------------------------------------------------------------
Why this is a SEPARATE module from consumption_plots.py
------------------------------------------------------------------------
Every function in consumption_plots.py operates on `T_matrix` — the
already-aggregated per-CLUSTER MEAN profile. Median and IQR are
statistics computed ACROSS HOUSEHOLDS, and T_matrix has already thrown
that per-household granularity away by the time it reaches
consumption_plots.py. This module instead takes the raw `wide`
DataFrame (index=timestamps, columns=household IDs — the same object
already used elsewhere in this codebase) plus the `labels` array, and
computes every statistic directly across the real households in each
cluster. Median/IQR (rather than mean/std) is used throughout because
it is robust to a small number of outlier households skewing the
picture.

Does NOT touch the clustering algorithm or recompute any reported
Sil/DBI/CHI metric — reads only the already-saved cluster_labels.npy
and the cached wide DataFrame.

Usage
-----
    from comprehensive_dashboard import generate_all_dashboard_figures
    from data_preprocessing import load_preprocessed
    import numpy as np

    _, wide = load_preprocessed(cache_dir, dataset)
    labels  = np.load(f"{results_dir}/cluster/K3/cluster_labels.npy")

    generate_all_dashboard_figures(wide, labels, dataset="london", K=3,
                                   output_dir=results_dir)
"""

import numpy as np
import pandas as pd
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple


# [NEW — journal-quality requirement] This module previously had no
# explicit style block at all (raw matplotlib defaults). Now matches
# generate_figures.py exactly, so every figure across the whole project
# shares one consistent visual style.
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


def cluster_label(cluster_id: int) -> str:
    """
    [MODIFIED — reviewer requirement] Returns the plain cluster label used
    in every legend, title, and axis annotation across this module.

    Previously this attached a semantic descriptor (e.g. "Cluster 0
    (Low)") inferred from average consumption. Removed per journal
    formatting requirement: cluster NUMBERING must reflect exactly what
    the clustering algorithm produced, with no implied consumption
    ordering — Cluster 2 is not guaranteed to be "High" just because it's
    the last index, and labelling it that way overstates a claim the
    numbering itself doesn't make. Every plotting function in this module
    now calls this single helper instead of hardcoding label strings, so
    there's exactly one place this convention lives.
    """
    return f"Cluster {cluster_id}"


def _detect_optimal_k(results_dir: str, dataset: str) -> int:
    """[NEW] Auto-detects the optimal K from metrics_all_k_<dataset>.csv's
    'Optimal' column — same logic as generate_figures.py, added here so K
    no longer has to be hardcoded/guessed when calling this module."""
    matches = glob.glob(str(Path(results_dir) / f"metrics_all_k_{dataset}.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No metrics_all_k_{dataset}.csv found in {results_dir} — "
            "either run the pipeline first, or pass K explicitly."
        )
    df = pd.read_csv(matches[0], index_col="K")
    optimal_rows = df[df["Optimal"] == True]  # noqa: E712
    if optimal_rows.empty:
        raise ValueError(f"No row marked Optimal=True in {matches[0]}")
    K = int(optimal_rows.index[0])
    print(f"[Dashboard] Auto-detected optimal K={K} for {dataset}")
    return K


def _cluster_color(k: int) -> str:
    return CLUSTER_COLORS[k % len(CLUSTER_COLORS)]


def _households_by_cluster(household_ids: List, labels: np.ndarray, K: int) -> dict:
    return {k: [h for h, lab in zip(household_ids, labels) if lab == k]
           for k in range(K)}


def _save_figure(fig: plt.Figure, filename_stem: str, dataset: str,
                 output_dir: str, use_tight_layout: bool = True) -> None:
    fig_dir = Path(output_dir) / "Figures" / "Dashboard" / dataset
    fig_dir.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass  # multi-axis figures with a shared colorbar (Fig D) can
                  # warn harmlessly here — bbox_inches="tight" below still
                  # produces a correctly-cropped save either way
    png_path = fig_dir / f"{filename_stem}.png"
    pdf_path = fig_dir / f"{filename_stem}.pdf"
    fig.savefig(str(png_path), dpi=600, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    print(f"[Dashboard] Saved -> {png_path}")
    print(f"[Dashboard] Saved -> {pdf_path}")
    plt.close(fig)


# ===========================================================================
# Fig A — Daily Consumption Trend (median + IQR)
# ===========================================================================

def plot_daily_trend_iqr(wide: pd.DataFrame, by_cluster: dict, K: int,
                         dataset: str, output_dir: str) -> None:
    daily_hh = wide.resample("D").sum()
    counts_per_day = wide.resample("D").count().iloc[:, 0]
    if len(counts_per_day) > 0 and counts_per_day.iloc[-1] < 48:
        daily_hh = daily_hh.iloc[:-1]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for k in range(K):
        hh = by_cluster[k]
        if not hh:
            continue
        sub = daily_hh[hh]
        median = sub.median(axis=1)
        q25, q75 = sub.quantile(0.25, axis=1), sub.quantile(0.75, axis=1)
        color = _cluster_color(k)
        ax.plot(median.index, median.values, color=color, linewidth=1.6,
               label=cluster_label(k), zorder=3)
        ax.fill_between(median.index, q25.values, q75.values,
                       color=color, alpha=0.18, zorder=2, linewidth=0)

    ax.set_title(f"Daily Consumption Trend (Median with IQR) — "
               f"{dataset.title()} K={K}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Consumption (kWh/day)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.text(0.5, -0.02,
            "Median (line) and IQR 25th-75th percentile (shaded) computed "
            "across households in each cluster. Incomplete final day excluded.",
            ha="center", fontsize=8, style="italic")
    _save_figure(fig, "Fig_A_DailyTrend", dataset, output_dir)


# ===========================================================================
# Fig B / C — Typical Daily Profile (original scale / shape-normalised)
# ===========================================================================

def _compute_daily_profile_stats(wide: pd.DataFrame, by_cluster: dict, K: int):
    """Shared computation for Fig B and Fig C — per-household mean profile
    across all days, then median/IQR across households. Computed ONCE and
    reused by both figures rather than recomputing."""
    unique_times = sorted(set(wide.index.time))
    profile_median, profile_q25, profile_q75 = {}, {}, {}
    for k in range(K):
        hh = by_cluster[k]
        if not hh:
            continue
        per_hh_profiles = wide[hh].groupby(wide.index.time).mean()
        profile_median[k] = per_hh_profiles.reindex(unique_times).median(axis=1)
        profile_q25[k]    = per_hh_profiles.reindex(unique_times).quantile(0.25, axis=1)
        profile_q75[k]    = per_hh_profiles.reindex(unique_times).quantile(0.75, axis=1)
    return unique_times, profile_median, profile_q25, profile_q75


def plot_typical_daily_profile(wide: pd.DataFrame, by_cluster: dict, K: int,
                               dataset: str, output_dir: str,
                               _precomputed: Optional[tuple] = None) -> tuple:
    """Fig_B_TypicalDailyProfile — original kWh/30min scale. Returns the
    precomputed stats tuple so plot_typical_daily_profile_normalized() can
    reuse it without recomputing."""
    stats = _precomputed or _compute_daily_profile_stats(wide, by_cluster, K)
    unique_times, profile_median, profile_q25, profile_q75 = stats

    x_ticks = list(range(len(unique_times)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in range(K):
        if k not in profile_median:
            continue
        color = _cluster_color(k)
        ax.plot(x_ticks, profile_median[k].values, color=color, linewidth=1.8,
               marker="o", markersize=2, label=cluster_label(k))
        ax.fill_between(x_ticks, profile_q25[k].values, profile_q75[k].values,
                       color=color, alpha=0.18, linewidth=0)

    step = max(1, len(unique_times) // 6)
    ax.set_xticks(x_ticks[::step])
    ax.set_xticklabels([unique_times[i].strftime("%H:%M") for i in x_ticks[::step]])
    ax.set_title(f"Typical Daily Profile (Median with IQR) — "
               f"{dataset.title()} K={K}\nOriginal Scale",
               fontsize=13, fontweight="bold")
    ax.set_xlabel("Time of Day"); ax.set_ylabel("Consumption (kWh/30min)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    _save_figure(fig, "Fig_B_TypicalDailyProfile", dataset, output_dir)
    return stats


def plot_typical_daily_profile_normalized(wide: pd.DataFrame, by_cluster: dict,
                                          K: int, dataset: str, output_dir: str,
                                          _precomputed: Optional[tuple] = None) -> None:
    """Fig_C_TypicalDailyProfileShape — shape-normalised (min-max on each
    cluster's median curve, IQR band rescaled by the same linear
    transform), 'for visualisation only'. Removes magnitude differences
    so peak TIMING can be compared directly across clusters."""
    stats = _precomputed or _compute_daily_profile_stats(wide, by_cluster, K)
    unique_times, profile_median, profile_q25, profile_q75 = stats
    x_ticks = list(range(len(unique_times)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in range(K):
        if k not in profile_median:
            continue
        color = _cluster_color(k)
        m = profile_median[k].values
        mn, mx = m.min(), m.max()
        scale = (mx - mn) if (mx - mn) > 1e-9 else 1.0
        m_norm    = (m - mn) / scale
        q25_norm  = (profile_q25[k].values - mn) / scale
        q75_norm  = (profile_q75[k].values - mn) / scale
        ax.plot(x_ticks, m_norm, color=color, linewidth=1.8,
               marker="o", markersize=2, label=cluster_label(k))
        ax.fill_between(x_ticks, q25_norm, q75_norm, color=color,
                       alpha=0.18, linewidth=0)

    step = max(1, len(unique_times) // 6)
    ax.set_xticks(x_ticks[::step])
    ax.set_xticklabels([unique_times[i].strftime("%H:%M") for i in x_ticks[::step]])
    ax.set_title(f"Typical Daily Profile (Median with IQR) — "
               f"{dataset.title()} K={K}\nShape-Normalised (for visualisation only)",
               fontsize=13, fontweight="bold")
    ax.set_xlabel("Time of Day"); ax.set_ylabel("Normalised Consumption (0-1)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    _save_figure(fig, "Fig_C_TypicalDailyProfileShape", dataset, output_dir)


# ===========================================================================
# Fig D — Date x Time-of-Day Heatmap (one sub-panel per cluster, ONE figure)
# ===========================================================================

def plot_datetime_heatmap(wide: pd.DataFrame, by_cluster: dict, K: int,
                          dataset: str, output_dir: str,
                          date_resample: str = "W") -> None:
    """Fig_D_DateTimeHeatmap — inherently a per-cluster COMPARISON, so all
    K clusters stay as sub-panels within this one standalone figure
    (matching the reference design), rather than being split further."""
    unique_times = sorted(set(wide.index.time))
    fig, axes = plt.subplots(1, K, figsize=(4.5 * K, 5), squeeze=False)
    axes = axes[0]

    heatmap_data, global_vmax = {}, 0.0
    for k in range(K):
        hh = by_cluster[k]
        if not hh:
            continue
        df_long = wide[hh].copy()
        df_long["_date_bucket"] = df_long.index.to_period(date_resample).astype(str)
        df_long["_time"] = df_long.index.time
        melted = df_long.melt(id_vars=["_date_bucket", "_time"],
                              value_vars=hh, value_name="kwh")
        pivot = melted.groupby(["_time", "_date_bucket"])["kwh"].median().unstack()
        pivot = pivot.reindex(unique_times)
        heatmap_data[k] = pivot
        if pivot.values.size:
            global_vmax = max(global_vmax, np.nanpercentile(pivot.values, 98))

    im = None
    for k in range(K):
        ax = axes[k]
        if k not in heatmap_data or heatmap_data[k].empty:
            ax.axis("off")
            continue
        pivot = heatmap_data[k]
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                      vmin=0, vmax=global_vmax, origin="upper")
        ax.set_title(cluster_label(k), fontsize=10)
        date_step = max(1, pivot.shape[1] // 4)
        ax.set_xticks(range(0, pivot.shape[1], date_step))
        ax.set_xticklabels(pivot.columns[::date_step], rotation=45,
                          fontsize=7, ha="right")
        ax.set_xlabel("Date", fontsize=8)
        if k == 0:
            time_step = max(1, len(unique_times) // 6)
            ax.set_yticks(range(0, len(unique_times), time_step))
            ax.set_yticklabels(
                [unique_times[i].strftime("%H:%M")
                for i in range(0, len(unique_times), time_step)], fontsize=7)
            ax.set_ylabel("Time of Day", fontsize=9)
        else:
            ax.set_yticks([])

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("kWh/30min (median)", fontsize=9)

    fig.suptitle(f"Date x Time-of-Day Heatmap (Median) — "
               f"{dataset.title()} K={K}", fontsize=14, fontweight="bold", y=1.03)
    _save_figure(fig, "Fig_D_DateTimeHeatmap", dataset, output_dir)


# ===========================================================================
# Fig E — Weekly Profile (median + IQR error bars)
# ===========================================================================

def plot_weekly_profile_iqr(wide: pd.DataFrame, by_cluster: dict, K: int,
                            dataset: str, output_dir: str) -> None:
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    width = 0.8 / K

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in range(K):
        hh = by_cluster[k]
        if not hh:
            continue
        per_hh_dow = wide[hh].groupby(wide.index.dayofweek).mean()
        per_hh_dow = per_hh_dow.reindex(range(7))
        med = per_hh_dow.median(axis=1)
        q25, q75 = per_hh_dow.quantile(0.25, axis=1), per_hh_dow.quantile(0.75, axis=1)
        color = _cluster_color(k)
        x_pos = np.arange(7) + (k - K / 2 + 0.5) * width
        ax.bar(x_pos, med.values, width=width * 0.9, color=color,
              label=cluster_label(k),
              yerr=[med.values - q25.values, q75.values - med.values],
              capsize=2, error_kw={"linewidth": 0.8, "alpha": 0.6})

    ax.set_title(f"Weekly Profile (Median with IQR) — "
               f"{dataset.title()} K={K}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Consumption (kWh/day)")
    ax.set_xticks(range(7)); ax.set_xticklabels(dow_names)
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    _save_figure(fig, "Fig_E_WeeklyProfile", dataset, output_dir)


# ===========================================================================
# Fig F — Monthly Profile (median + IQR band)
# ===========================================================================

def plot_monthly_profile_iqr(wide: pd.DataFrame, by_cluster: dict, K: int,
                             dataset: str, output_dir: str) -> None:
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k in range(K):
        hh = by_cluster[k]
        if not hh:
            continue
        per_hh_month = wide[hh].groupby(wide.index.month).mean().reindex(range(1, 13))
        med = per_hh_month.median(axis=1)
        q25, q75 = per_hh_month.quantile(0.25, axis=1), per_hh_month.quantile(0.75, axis=1)
        color = _cluster_color(k)
        ax.plot(range(12), med.values, "o-", color=color, linewidth=1.8,
               markersize=3, label=cluster_label(k))
        ax.fill_between(range(12), q25.values, q75.values, color=color,
                       alpha=0.18, linewidth=0)

    ax.set_title(f"Monthly Profile (Median with IQR) — "
               f"{dataset.title()} K={K}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Consumption (kWh/day)")
    ax.set_xticks(range(12)); ax.set_xticklabels(month_names)
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.3)
    _save_figure(fig, "Fig_F_MonthlyProfile", dataset, output_dir)


# ===========================================================================
# Fig G — Seasonal Profile (median + IQR error bars)
# ===========================================================================

def plot_seasonal_profile_iqr(wide: pd.DataFrame, by_cluster: dict, K: int,
                              dataset: str, output_dir: str) -> None:
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                 3: "Spring", 4: "Spring", 5: "Spring",
                 6: "Summer", 7: "Summer", 8: "Summer",
                 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_of_month = pd.Series(wide.index.month).map(season_map).values
    seasons_idx = pd.Categorical(season_of_month, categories=season_order, ordered=True)
    width = 0.8 / K

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in range(K):
        hh = by_cluster[k]
        if not hh:
            continue
        per_hh_season = wide[hh].groupby(seasons_idx).mean().reindex(season_order)
        med = per_hh_season.median(axis=1)
        q25, q75 = per_hh_season.quantile(0.25, axis=1), per_hh_season.quantile(0.75, axis=1)
        color = _cluster_color(k)
        x_pos = np.arange(4) + (k - K / 2 + 0.5) * width
        ax.bar(x_pos, med.values, width=width * 0.9, color=color,
              label=cluster_label(k),
              yerr=[med.values - q25.values, q75.values - med.values],
              capsize=2, error_kw={"linewidth": 0.8, "alpha": 0.6})

    ax.set_title(f"Seasonal Profile (Median with IQR) — "
               f"{dataset.title()} K={K}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season"); ax.set_ylabel("Consumption (kWh/day)")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{s}\n({m})" for s, m in zip(
        season_order, ["Dec-Feb", "Mar-May", "Jun-Aug", "Sep-Nov"])])
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    _save_figure(fig, "Fig_G_SeasonalProfile", dataset, output_dir)


# ===========================================================================
# Orchestrator
# ===========================================================================

def generate_all_dashboard_figures(wide: pd.DataFrame, labels: np.ndarray,
                                   dataset: str, output_dir: str,
                                   K: Optional[int] = None,
                                   heatmap_date_resample: str = "W") -> None:
    """Generates all 7 figures (A-G) as separate PNG+PDF files, reusing the
    shared daily-profile computation (Fig B/C) once rather than twice.

    K is optional here for backward compatibility, but NOTE: if you
    already loaded `labels` from a specific K's cluster_labels.npy
    yourself, you MUST pass that same K explicitly — auto-detecting K
    here cannot retroactively fix a `labels` array that was already
    loaded for the wrong K. For a fully safe, auto-detecting call that
    loads everything itself, use generate_dashboard_from_results() instead.
    """
    if K is None:
        K = _detect_optimal_k(output_dir, dataset)

    household_ids = list(wide.columns[:len(labels)])
    by_cluster = _households_by_cluster(household_ids, labels, K)

    print(f"\n{'='*60}\nGenerating 7 dashboard figures — "
        f"{dataset.upper()} K={K}\n{'='*60}")

    plot_daily_trend_iqr(wide, by_cluster, K, dataset, output_dir)
    stats = plot_typical_daily_profile(wide, by_cluster, K, dataset, output_dir)
    plot_typical_daily_profile_normalized(wide, by_cluster, K, dataset,
                                          output_dir, _precomputed=stats)
    plot_datetime_heatmap(wide, by_cluster, K, dataset, output_dir,
                          date_resample=heatmap_date_resample)
    plot_weekly_profile_iqr(wide, by_cluster, K, dataset, output_dir)
    plot_monthly_profile_iqr(wide, by_cluster, K, dataset, output_dir)
    plot_seasonal_profile_iqr(wide, by_cluster, K, dataset, output_dir)

    print(f"\n[Dashboard] All 7 figures saved to "
        f"{output_dir}/Figures/Dashboard/{dataset}/")


def generate_dashboard_from_results(results_dir: str, cache_dir: str,
                                    dataset: Optional[str] = None,
                                    K: Optional[int] = None,
                                    heatmap_date_resample: str = "W") -> None:
    """
    [NEW] Fully self-contained entry point — loads `wide` and the CORRECT
    K's `labels` itself (auto-detecting both dataset and optimal K from
    metrics_all_k_<dataset>.csv if not given), so there is no way for the
    caller to accidentally load labels for one K while requesting figures
    for another. This is the RECOMMENDED way to call this module — mirrors
    generate_figures.py's load_run_artifacts() pattern exactly.

    Does NOT retrain or re-cluster — reads only already-saved artifacts.
    """
    import glob as _glob
    from data_preprocessing import load_preprocessed

    if dataset is None or K is None:
        if dataset is None:
            matches = _glob.glob(str(Path(results_dir) / "metrics_all_k_*.csv"))
            if not matches:
                raise FileNotFoundError(
                    f"No metrics_all_k_*.csv found in {results_dir}.")
            dataset = Path(matches[0]).stem.replace("metrics_all_k_", "")
        if K is None:
            K = _detect_optimal_k(results_dir, dataset)

    _, wide = load_preprocessed(cache_dir, dataset)
    if wide is None:
        raise FileNotFoundError(f"No cached data for {dataset} in {cache_dir}.")

    labels_path = Path(results_dir) / "cluster" / f"K{K}" / "cluster_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")
    labels = np.load(str(labels_path))

    print(f"[Dashboard] Loaded {dataset} K={K}: wide={wide.shape}, "
          f"labels={labels.shape}")

    generate_all_dashboard_figures(wide, labels, dataset=dataset, K=K,
                                   output_dir=results_dir,
                                   heatmap_date_resample=heatmap_date_resample)
