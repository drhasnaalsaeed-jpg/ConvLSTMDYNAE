"""
consumption_plots.py
====================
Comprehensive consumption pattern visualisation for ConvLSTM-DynAE results.

Generates six plot types per dataset, all saved to your results folder:

  1. Daily consumption        — mean kWh per day across the full period
  2. Intraday (24-h) profile  — mean half-hourly consumption within a day
  3. Weekly profile           — mean consumption by day-of-week (Mon–Sun)
  4. Monthly consumption      — mean consumption per calendar month
  5. Seasonal consumption     — mean consumption per season (4 seasons)
  6. Cluster comparison grid  — all 5 views side-by-side per cluster

Usage (standalone)
------------------
  python consumption_plots.py \
      --profiles_csv  results/london/load_profiles_halfhourly.csv \
      --output_dir    results/london/consumption_plots

Usage (from main.py / Colab)
-----------------------------
  from consumption_plots import plot_all_consumption_patterns
  plot_all_consumption_patterns(T_matrix, output_dir="results/london/consumption_plots")
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional


# [NEW — journal-quality requirement] This module previously had no
# explicit style block at all (raw matplotlib defaults). Now matches
# generate_figures.py / comprehensive_dashboard.py exactly, so every
# figure across the whole project shares one consistent visual style.
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


# ---------------------------------------------------------------------------
# Colour palette — one colour per cluster, consistent across all plots
# ---------------------------------------------------------------------------
CLUSTER_COLORS = ["#2196F3", "#E53935", "#43A047", "#FB8C00",
                  "#8E24AA", "#00ACC1", "#F4511E", "#6D4C41", "#546E7A"]

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
     3: "Spring", 4: "Spring", 5: "Spring",
     6: "Summer", 7: "Summer", 8: "Summer",
     9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER  = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_COLORS = {"Winter": "#5C9BD6", "Spring": "#81C784",
                 "Summer": "#FFB74D", "Autumn": "#A1887F"}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Helper: save figure
# ---------------------------------------------------------------------------

def _save(fig, path: str):
    """[MODIFIED — journal-quality requirement] Now saves BOTH 600dpi PNG
    and vector PDF (was PNG-only at 150dpi), matching generate_figures.py
    and comprehensive_dashboard.py."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    pdf_path = str(Path(path).with_suffix(".pdf"))
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  [Saved] {path}")
    print(f"  [Saved] {pdf_path}")
    plt.close(fig)


# ===========================================================================
# 1. DAILY CONSUMPTION
# ===========================================================================

def plot_daily_consumption(T_matrix:  pd.DataFrame,
                           title:     str = "Daily Consumption by Cluster",
                           save_path: Optional[str] = None,
                           rolling_window: int = 7) -> plt.Figure:
    """
    Mean daily consumption (kWh/day) for each cluster over the full period.
    X-axis = calendar date, Y-axis = normalised consumption.

    [MODIFIED] The raw daily series is now shown thin/faint (so the real
    weekday/weekend weekly cycle is still visible for anyone who wants
    it), with a BOLD `rolling_window`-day rolling mean drawn on top plus
    a shaded ±1 rolling-std confidence band. This directly addresses the
    case where two clusters' raw daily lines cross and tangle so much
    they're hard to visually compare — the smoothed trend + band makes
    it immediately clear whether the clusters are genuinely distinct at
    the seasonal timescale or genuinely overlapping, rather than just
    looking noisy. rolling_window=7 fully cancels the weekly cycle
    (divides it out exactly) while preserving the seasonal arc.
    """
    daily = T_matrix.resample("D").mean()
    rolling_mean = daily.rolling(window=rolling_window, center=True,
                                 min_periods=1).mean()
    rolling_std  = daily.rolling(window=rolling_window, center=True,
                                 min_periods=1).std()

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, col in enumerate(daily.columns):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        # Raw daily data — thin and faint, still visible on request but
        # no longer competing visually with the trend
        ax.plot(daily.index, daily[col], color=color, linewidth=0.5,
                alpha=0.25, zorder=1)

        # Bold rolling-mean trend line — this is what the eye should
        # actually track for cluster comparison
        ax.plot(rolling_mean.index, rolling_mean[col], color=color,
                linewidth=2.2, alpha=0.95, label=col, zorder=3)

        # Confidence band (±1 rolling std) — makes genuine overlap vs
        # genuine separation immediately visible, rather than implied
        upper = rolling_mean[col] + rolling_std[col]
        lower = rolling_mean[col] - rolling_std[col]
        ax.fill_between(rolling_mean.index, lower, upper,
                        color=color, alpha=0.12, zorder=2, linewidth=0)

    ax.set_title(f"{title}\n"
               f"(thin lines = raw daily data, bold = {rolling_window}-day "
               f"rolling mean, shaded = ±1 rolling std)",
               fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Normalised Consumption (kWh/day)", fontsize=11)
    ax.legend(title="Cluster", fontsize=9, title_fontsize=9, loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Shade weekends lightly (kept from the original — still meaningful
    # context even with the trend line added)
    for date in daily.index:
        if date.weekday() >= 5:
            ax.axvspan(date, date + pd.Timedelta(days=1),
                       alpha=0.04, color="gray", linewidth=0)

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ===========================================================================
# 2. INTRADAY (24-HOUR) CONSUMPTION PROFILE
# ===========================================================================

def plot_intraday_consumption(T_matrix:  pd.DataFrame,
                              title:     str = "Mean Intraday Consumption Profile (24h)",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Average consumption for each 30-minute slot across the full dataset.
    X-axis = time of day (00:00 – 23:30), Y-axis = mean normalised consumption.
    Shows the typical 24-hour consumption shape per cluster.
    """
    # Group by time-of-day (hour + minute)
    intraday = T_matrix.copy()
    intraday["time"] = intraday.index.time
    intraday_mean = intraday.groupby("time")[T_matrix.columns].mean()

    # Build a readable x-axis: 48 half-hour labels
    time_labels = [f"{h:02d}:{m:02d}"
                   for h in range(24) for m in (0, 30)]
    x = np.arange(len(time_labels))

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, col in enumerate(T_matrix.columns):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.plot(x, intraday_mean[col].values,
                label=col, color=color, linewidth=2.0)
        ax.fill_between(x, intraday_mean[col].values,
                        alpha=0.10, color=color)

    # Tick every 2 hours (every 4th half-hour slot)
    tick_pos   = x[::4]
    tick_label = [time_labels[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_label, rotation=45, fontsize=8)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Time of Day", fontsize=11)
    ax.set_ylabel("Mean Normalised Consumption", fontsize=11)
    ax.legend(title="Cluster", fontsize=9, title_fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Shade night hours (22:00–06:00)
    ax.axvspan(0,  12, alpha=0.05, color="navy",  label="_nolegend_")
    ax.axvspan(44, 48, alpha=0.05, color="navy",  label="_nolegend_")

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ===========================================================================
# 3. WEEKLY CONSUMPTION PROFILE
# ===========================================================================

def plot_weekly_consumption(T_matrix:  pd.DataFrame,
                            title:     str = "Mean Weekly Consumption Profile",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Mean consumption for each day of the week (Monday=0 … Sunday=6).
    Reveals weekday vs weekend behaviour per cluster.
    """
    weekly = T_matrix.copy()
    weekly["dayofweek"] = weekly.index.dayofweek
    weekly_mean = weekly.groupby("dayofweek")[T_matrix.columns].mean()

    x     = np.arange(7)
    width = 0.8 / len(T_matrix.columns)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, col in enumerate(T_matrix.columns):
        color  = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        offset = (i - len(T_matrix.columns) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, weekly_mean[col].values,
                        width=width * 0.9, label=col,
                        color=color, alpha=0.82, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(DAY_NAMES, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Mean Normalised Consumption", fontsize=11)
    ax.legend(title="Cluster", fontsize=9, title_fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Highlight weekend
    ax.axvspan(4.5, 6.5, alpha=0.07, color="orange", label="_nolegend_")
    ax.text(5.5, ax.get_ylim()[1] * 0.97, "Weekend",
            ha="center", va="top", fontsize=8, color="darkorange", alpha=0.7)

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ===========================================================================
# 4. MONTHLY CONSUMPTION
# ===========================================================================

def plot_monthly_consumption(T_matrix:  pd.DataFrame,
                             title:     str = "Mean Monthly Consumption",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Mean consumption aggregated per calendar month (Jan–Dec).
    Reveals seasonal drift and annual patterns.
    """
    monthly = T_matrix.copy()
    monthly["month"] = monthly.index.month
    monthly_mean = monthly.groupby("month")[T_matrix.columns].mean()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x     = np.arange(12)
    width = 0.8 / len(T_matrix.columns)

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, col in enumerate(T_matrix.columns):
        color  = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        offset = (i - len(T_matrix.columns) / 2 + 0.5) * width
        ax.bar(x + offset, monthly_mean.reindex(range(1, 13))[col].values,
               width=width * 0.9, label=col,
               color=color, alpha=0.82, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(month_names, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Mean Normalised Consumption", fontsize=11)
    ax.legend(title="Cluster", fontsize=9, title_fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Shade seasons lightly
    season_spans = [(0, 2, "Winter", "#90CAF9"),
                    (2, 5, "Spring", "#A5D6A7"),
                    (5, 8, "Summer", "#FFE082"),
                    (8, 11, "Autumn", "#BCAAA4")]
    ymax = ax.get_ylim()[1]
    for s, e, name, col in season_spans:
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.12, color=col, label="_nolegend_")
        ax.text((s + e) / 2, ymax * 0.98, name,
                ha="center", va="top", fontsize=8, color="dimgray")

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ===========================================================================
# 5. SEASONAL CONSUMPTION
# ===========================================================================

def plot_seasonal_consumption(T_matrix:  pd.DataFrame,
                              title:     str = "Mean Seasonal Consumption",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Two complementary views in one figure:
      Top    — bar chart: mean consumption per season per cluster
      Bottom — intraday profile per season (averaged across all clusters)
    """
    seasonal = T_matrix.copy()
    seasonal["season"] = seasonal.index.month.map(SEASON_MAP)

    season_mean  = seasonal.groupby("season")[T_matrix.columns].mean()
    season_mean  = season_mean.reindex(SEASON_ORDER)

    # Intraday per season (all clusters averaged)
    seasonal["time"] = seasonal.index.time
    intraday_season  = (seasonal.groupby(["season", "time"])[T_matrix.columns]
                        .mean()
                        .mean(axis=1)          # average across clusters
                        .unstack(level=0))     # columns = seasons

    time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    x48 = np.arange(48)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10),
                             gridspec_kw={"height_ratios": [1, 1.2]})
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    # ── Top: seasonal bar chart ──────────────────────────────────
    ax1   = axes[0]
    n_cl  = len(T_matrix.columns)
    width = 0.7 / n_cl
    x4    = np.arange(4)

    for i, col in enumerate(T_matrix.columns):
        color  = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        offset = (i - n_cl / 2 + 0.5) * width
        ax1.bar(x4 + offset,
                season_mean[col].values,
                width=width * 0.9, label=col,
                color=color, alpha=0.85, edgecolor="white")

    ax1.set_xticks(x4)
    ax1.set_xticklabels(SEASON_ORDER, fontsize=11)
    ax1.set_ylabel("Mean Normalised Consumption", fontsize=10)
    ax1.set_title("(a) Mean consumption per season per cluster",
                  fontsize=11, pad=8)
    ax1.legend(title="Cluster", fontsize=9, title_fontsize=9)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Colour background per season
    for xi, season in enumerate(SEASON_ORDER):
        ax1.get_xticklabels()[xi].set_color(
            {"Winter": "#1565C0", "Spring": "#2E7D32",
             "Summer": "#E65100", "Autumn": "#4E342E"}[season]
        )

    # ── Bottom: intraday by season ───────────────────────────────
    ax2 = axes[1]
    for season in SEASON_ORDER:
        if season in intraday_season.columns:
            color = SEASON_COLORS[season]
            vals  = intraday_season[season].values
            ax2.plot(x48, vals, label=season, color=color, linewidth=2.2)
            ax2.fill_between(x48, vals, alpha=0.10, color=color)

    tick_pos   = x48[::4]
    tick_label = [time_labels[i] for i in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_label, rotation=45, fontsize=8)
    ax2.set_xlabel("Time of Day", fontsize=11)
    ax2.set_ylabel("Mean Normalised Consumption", fontsize=10)
    ax2.set_title("(b) Mean intraday profile per season (all clusters)",
                  fontsize=11, pad=8)
    ax2.legend(title="Season", fontsize=9, title_fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ===========================================================================
# 6. CLUSTER COMPARISON GRID (all views in one figure per cluster)
# ===========================================================================

def plot_cluster_comparison_grid(T_matrix:  pd.DataFrame,
                                 title:     str = "Cluster Consumption Comparison",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    One row per cluster, five columns:
      Col 1 — Daily trend
      Col 2 — Intraday (24h) profile
      Col 3 — Weekly profile
      Col 4 — Monthly profile
      Col 5 — Seasonal profile
    """
    clusters = T_matrix.columns.tolist()
    K        = len(clusters)

    # Pre-compute aggregates
    daily_df    = T_matrix.resample("D").mean()

    intra       = T_matrix.copy()
    intra["t"]  = intra.index.time
    intra_mean  = intra.groupby("t")[clusters].mean()

    weekly      = T_matrix.copy()
    weekly["d"] = weekly.index.dayofweek
    week_mean   = weekly.groupby("d")[clusters].mean()

    monthly     = T_matrix.copy()
    monthly["m"]= monthly.index.month
    month_mean  = monthly.groupby("m")[clusters].mean().reindex(range(1, 13))

    seasonal    = T_matrix.copy()
    seasonal["s"] = seasonal.index.month.map(SEASON_MAP)
    seas_mean   = seasonal.groupby("s")[clusters].mean().reindex(SEASON_ORDER)

    time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    x48 = np.arange(48)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(K, 5,
                             figsize=(22, 4 * K),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.5})

    # Handle K=1 edge case
    if K == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Daily Trend", "24h Intraday",
                  "Weekly", "Monthly", "Seasonal"]

    for ci, cluster in enumerate(clusters):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        row   = axes[ci]

        # Row label
        row[0].set_ylabel(cluster, fontsize=11,
                          fontweight="bold", color=color, labelpad=12)

        # Column headers only on first row
        if ci == 0:
            for j, ct in enumerate(col_titles):
                row[j].set_title(ct, fontsize=11, fontweight="bold", pad=8)

        # ── Col 0: Daily ─────────────────────────────────────────
        ax = row[0]
        ax.plot(daily_df.index, daily_df[cluster],
                color=color, linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Consumption", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)

        # ── Col 1: Intraday ───────────────────────────────────────
        ax = row[1]
        vals = intra_mean[cluster].values
        ax.plot(x48, vals, color=color, linewidth=2.0)
        ax.fill_between(x48, vals, alpha=0.15, color=color)
        ax.set_xticks(x48[::8])
        ax.set_xticklabels([time_labels[i] for i in x48[::8]],
                            fontsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Time of day", fontsize=8)
        ax.set_ylabel("Consumption", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

        # ── Col 2: Weekly ─────────────────────────────────────────
        ax = row[2]
        ax.bar(range(7), week_mean[cluster].values,
               color=color, alpha=0.82, edgecolor="white")
        ax.set_xticks(range(7))
        ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                            fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Day of week", fontsize=8)
        ax.set_ylabel("Consumption", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.axvspan(4.5, 6.5, alpha=0.10, color="orange")

        # ── Col 3: Monthly ────────────────────────────────────────
        ax = row[3]
        ax.bar(range(12), month_mean[cluster].values,
               color=color, alpha=0.82, edgecolor="white")
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names, fontsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Month", fontsize=8)
        ax.set_ylabel("Consumption", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # ── Col 4: Seasonal ───────────────────────────────────────
        ax = row[4]
        sea_vals = seas_mean[cluster].values
        bar_colors = [SEASON_COLORS[s] for s in SEASON_ORDER]
        bars = ax.bar(range(4), sea_vals,
                      color=bar_colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(4))
        ax.set_xticklabels(SEASON_ORDER, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Season", fontsize=8)
        ax.set_ylabel("Consumption", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Value labels on top of seasonal bars
        for bar, val in zip(bars, sea_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# ===========================================================================
# Master function — run all plots at once
# ===========================================================================

def plot_all_consumption_patterns(T_matrix:   pd.DataFrame,
                                  output_dir: str = "results/consumption_plots",
                                  dataset:    str = "dataset"):
    """
    Generate and save all six plot types.

    Parameters
    ----------
    T_matrix   : half-hourly load profile DataFrame
                 (index = DatetimeIndex, columns = cluster names)
    output_dir : folder where PNGs are saved
    dataset    : label for plot titles and filenames ('london' or 'irish')
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ds = dataset.title()

    print(f"\n{'='*60}")
    print(f"Generating consumption plots → {output_dir}/")
    print(f"{'='*60}")

    print("\n[1/6] Daily consumption …")
    plot_daily_consumption(
        T_matrix,
        title     = f"Daily Consumption — {ds} Dataset",
        save_path = f"{output_dir}/01_daily_consumption_{dataset}.png",
    )

    print("[2/6] Intraday (24h) profile …")
    plot_intraday_consumption(
        T_matrix,
        title     = f"Intraday Consumption Profile (24h) — {ds} Dataset",
        save_path = f"{output_dir}/02_intraday_consumption_{dataset}.png",
    )

    print("[3/6] Weekly profile …")
    plot_weekly_consumption(
        T_matrix,
        title     = f"Weekly Consumption Profile — {ds} Dataset",
        save_path = f"{output_dir}/03_weekly_consumption_{dataset}.png",
    )

    print("[4/6] Monthly consumption …")
    plot_monthly_consumption(
        T_matrix,
        title     = f"Monthly Consumption — {ds} Dataset",
        save_path = f"{output_dir}/04_monthly_consumption_{dataset}.png",
    )

    print("[5/6] Seasonal consumption …")
    plot_seasonal_consumption(
        T_matrix,
        title     = f"Seasonal Consumption — {ds} Dataset",
        save_path = f"{output_dir}/05_seasonal_consumption_{dataset}.png",
    )

    print("[6/6] Cluster comparison grid …")
    plot_cluster_comparison_grid(
        T_matrix,
        title     = f"Cluster Consumption Comparison — {ds} Dataset",
        save_path = f"{output_dir}/06_cluster_comparison_grid_{dataset}.png",
    )

    print(f"\n✓ All 6 plots saved to: {output_dir}/")
    print("  01_daily_consumption")
    print("  02_intraday_consumption  (24h)")
    print("  03_weekly_consumption")
    print("  04_monthly_consumption")
    print("  05_seasonal_consumption")
    print("  06_cluster_comparison_grid")


# ===========================================================================
# Display all saved plots inline (for Colab)
# ===========================================================================

def display_all_plots_colab(output_dir: str):
    """
    Call this in a Colab cell to display all saved PNGs inline.
    """
    import matplotlib.image as mpimg
    import glob

    plot_files = sorted(glob.glob(f"{output_dir}/*.png"))
    if not plot_files:
        print(f"No PNG files found in {output_dir}")
        return

    for fpath in plot_files:
        fname = Path(fpath).stem.replace("_", " ").title()
        print(f"\n── {fname} ──")
        img = mpimg.imread(fpath)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(img)
        ax.axis("off")
        plt.tight_layout()
        plt.show()


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate consumption pattern plots from load profile CSV"
    )
    p.add_argument("--profiles_csv", required=True,
                   help="Path to load_profiles_halfhourly.csv")
    p.add_argument("--output_dir",   default="results/consumption_plots",
                   help="Folder to save PNG files")
    p.add_argument("--dataset",      default="dataset",
                   choices=["london", "irish", "dataset"],
                   help="Dataset label used in plot titles")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading profiles from: {args.profiles_csv}")
    T_matrix = pd.read_csv(args.profiles_csv, index_col=0, parse_dates=True)
    print(f"Profile shape: {T_matrix.shape}  "
          f"({T_matrix.shape[0]} timestamps × {T_matrix.shape[1]} clusters)")

    plot_all_consumption_patterns(
        T_matrix   = T_matrix,
        output_dir = args.output_dir,
        dataset    = args.dataset,
    )
