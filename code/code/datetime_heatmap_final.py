"""
Publication-quality Date x Time-of-Day heatmap for Scientific Reports.

Two pieces, kept deliberately separate:

  1. compute_cluster_heatmaps() - EXACT same data computation as the
     original plot_datetime_heatmap() in comprehensive_dashboard.py
     (median kWh/30min per time-of-day x date-bucket cell, per cluster).
     The only change vs. the original: date-bucket columns are aligned
     onto ONE shared axis across all K clusters (so every panel has
     identical x-limits, as required for a fair side-by-side comparison),
     and kept as real timestamps instead of strings (needed for "Jul 2009"
     -style tick formatting). The computed median VALUES are untouched.

  2. plot_datetime_heatmap_final() - the new publication-quality plotting
     function. Pure presentation layer - takes already-computed heatmaps
     in, does not recompute or alter any value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# ===========================================================================
# 1) DATA — unchanged computation, just shared-axis alignment
# ===========================================================================

def compute_cluster_heatmaps(wide: pd.DataFrame, by_cluster: dict, K: int,
                              date_resample: str = "W"):
    """
    Reproduces the original plot_datetime_heatmap() pivot computation
    exactly (median kWh/30min per [time-of-day, date-bucket] cell),
    then aligns every cluster's pivot onto one shared, sorted date axis.

    Returns
    -------
    heatmaps    : list[np.ndarray], length K, each shape
                  (len(time_labels), len(dates)). NaN where a cluster has
                  no data for a given date bucket.
    dates       : list[pd.Timestamp] - shared x-axis, one per column.
    time_labels : list[datetime.time] - shared y-axis, one per row.
    """
    unique_times = sorted(set(wide.index.time))

    per_cluster_pivot = {}
    all_date_buckets = set()
    for k in range(K):
        hh = by_cluster.get(k, [])
        if not hh:
            per_cluster_pivot[k] = None
            continue
        df_long = wide[hh].copy()
        # Period objects (not .astype(str)) so real dates survive through
        # to the plotting step for "Jul 2009"-style tick formatting.
        df_long["_date_bucket"] = df_long.index.to_period(date_resample)
        df_long["_time"] = df_long.index.time
        melted = df_long.melt(id_vars=["_date_bucket", "_time"],
                              value_vars=hh, value_name="kwh")
        pivot = melted.groupby(["_time", "_date_bucket"])["kwh"].median().unstack()
        pivot = pivot.reindex(unique_times)
        per_cluster_pivot[k] = pivot
        all_date_buckets.update(pivot.columns)

    shared_dates = sorted(all_date_buckets)

    heatmaps = []
    for k in range(K):
        pivot = per_cluster_pivot[k]
        if pivot is None:
            heatmaps.append(np.full((len(unique_times), len(shared_dates)), np.nan))
        else:
            heatmaps.append(pivot.reindex(columns=shared_dates).values)

    time_labels = unique_times
    dates = [p.start_time for p in shared_dates]
    return heatmaps, dates, time_labels


# ===========================================================================
# 2) PLOT — publication-quality, presentation only
# ===========================================================================

def plot_datetime_heatmap_final(heatmaps, dates, time_labels, cluster_names,
                                 output_filename,
                                 vmin=None, vmax=None,
                                 use_percentile_clip=True,
                                 cmap="viridis",
                                 caption_dataset_name=None):
    """
    Publication-quality Date x Time-of-Day heatmap, Scientific Reports
    layout. Does not compute or modify heatmap values - pass in the output
    of compute_cluster_heatmaps() (or any (time x date) arrays sharing the
    same date/time axes).

    Parameters
    ----------
    heatmaps       : list[np.ndarray], one per cluster, shape
                      (len(time_labels), len(dates)).
    dates          : shared x-axis values (list[pd.Timestamp]-like), one
                      per column of every heatmap.
    time_labels    : shared y-axis values (list[datetime.time]), one per
                      row of every heatmap.
    cluster_names  : list[str], len == len(heatmaps). Used ONLY as panel
                      titles ("Cluster 0", "Cluster 1", ...) - no dataset
                      name or K is drawn on the figure.
    output_filename: path to save the PNG (dpi=600, white background).
    vmin, vmax     : shared color-scale bounds, computed ONCE across all
                      panels (never per-panel). If both None: 1st/99th
                      percentile if use_percentile_clip else true min/max.
    caption_dataset_name : e.g. "London" or "Irish" - NOT drawn on the
                      figure; folded into the returned caption text only,
                      per the spec's "dataset name in caption, not title".

    Returns
    -------
    fig     : matplotlib Figure (already saved to output_filename).
    caption : str, a ready-to-use figure caption mentioning the dataset.
    """
    n_panels = len(heatmaps)
    assert len(cluster_names) == n_panels, \
        "cluster_names must have exactly one entry per heatmap"

    # ---- ONE shared color scale across all panels (never normalised per-panel) ----
    if vmin is None or vmax is None:
        all_vals = np.concatenate([h[~np.isnan(h)].ravel() for h in heatmaps])
        if use_percentile_clip:
            vmin = float(np.percentile(all_vals, 1)) if vmin is None else vmin
            vmax = float(np.percentile(all_vals, 99)) if vmax is None else vmax
        else:
            vmin = float(np.nanmin(all_vals)) if vmin is None else vmin
            vmax = float(np.nanmax(all_vals)) if vmax is None else vmax

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")   # missing date buckets render white, not a stray colour

    # ---- N panels + one dedicated narrow colorbar column ----
    fig_w = float(np.clip(4.7 * n_panels + 2.0, 16, 18))
    width_ratios = [1] * n_panels + [0.06]
    fig, axes = plt.subplots(
        1, n_panels + 1, figsize=(fig_w, 5.5),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.15},
    )
    panel_axes, cbar_ax = list(axes[:-1]), axes[-1]
    fig.patch.set_facecolor("white")

    n_dates = len(dates)
    n_time = len(time_labels)

    im = None
    for i, (ax, hm, name) in enumerate(zip(panel_axes, heatmaps, cluster_names)):
        im = ax.imshow(hm, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest")
        ax.set_title(name, fontsize=14, fontweight="bold")

        # X axis: real dates, ~5-6 ticks, "Jul 2009" style, rotated 30 deg
        n_ticks = min(6, n_dates)
        tick_idx = sorted(set(np.linspace(0, n_dates - 1, n_ticks).round().astype(int)))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([pd.Timestamp(dates[j]).strftime("%b %Y") for j in tick_idx],
                           rotation=30, ha="right", fontsize=10)
        ax.set_xlabel("Date", fontsize=12)

        # Y axis: "Time of day" label + ticks on the FIRST panel only
        if i == 0:
            t_step = max(1, n_time // 6)
            t_idx = list(range(0, n_time, t_step))
            ax.set_yticks(t_idx)
            ax.set_yticklabels(
                [time_labels[j].strftime("%H:%M") for j in t_idx], fontsize=10)
            ax.set_ylabel("Time of day", fontsize=12)
        else:
            ax.set_yticks([])

        # identical axis limits across all panels
        ax.set_xlim(-0.5, n_dates - 0.5)
        ax.set_ylim(n_time - 0.5, -0.5)
        ax.tick_params(axis="both", labelsize=10)

    # ---- ONE shared colorbar, dedicated axis, never overlapping a panel ----
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Median consumption (kWh/30 min)", fontsize=12, labelpad=14)
    cbar.ax.tick_params(labelsize=10)

    # Dataset name intentionally NOT in the title - caption-only, per spec.
    fig.suptitle("Date\u2013Time Heatmaps of Median Electricity Consumption",
                fontsize=17.5, fontweight="bold", y=1.03)

    fig.subplots_adjust(bottom=0.22, top=0.86)
    fig.savefig(output_filename, dpi=600, facecolor="white", bbox_inches="tight")

    scale_note = "1st-99th percentile clipped" if use_percentile_clip else "min-max"
    dataset_note = f" for the {caption_dataset_name} dataset" if caption_dataset_name else ""
    caption = (
        f"Date\u2013time heatmaps of median half-hourly electricity consumption "
        f"for each discovered cluster{dataset_note} (K={n_panels}). "
        f"Color scale ({scale_note}) is shared across all panels to allow "
        f"direct comparison of absolute consumption between clusters."
    )
    return fig, caption


# ===========================================================================
# 3) DRIVER — generates both London and Irish figures from saved artifacts
# ===========================================================================

def generate_both_datetime_heatmaps(results_base_dir, cache_dir, output_dir="."):
    """
    Loads each dataset's saved cluster artifacts (no retraining, no
    re-clustering - same artifact-loading approach as generate_figures.py's
    load_run_artifacts), builds the heatmaps via compute_cluster_heatmaps(),
    and renders both final figures via plot_datetime_heatmap_final().

    results_base_dir : e.g. f"{RESULTS}"  containing london/ and irish/
    cache_dir         : e.g. f"{DATA}/cache"
    output_dir        : where London_DateTimeHeatmap_Final.png and
                         Irish_DateTimeHeatmap_Final.png get written
    """
    from generate_figures import load_run_artifacts
    from comprehensive_dashboard import _households_by_cluster

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, out_name in [("london", "London_DateTimeHeatmap_Final.png"),
                              ("irish",  "Irish_DateTimeHeatmap_Final.png")]:
        print(f"\n[{dataset.upper()}] Loading saved artifacts …")
        art = load_run_artifacts(
            results_dir=f"{results_base_dir}/{dataset}",
            cache_dir=cache_dir, dataset=dataset)

        wide, labels, K = art["wide"], art["labels"], art["K"]
        household_ids = list(wide.columns)
        by_cluster = _households_by_cluster(household_ids, labels, K)

        heatmaps, dates, time_labels = compute_cluster_heatmaps(wide, by_cluster, K)
        cluster_names = [f"Cluster {k}" for k in range(K)]

        fig, caption = plot_datetime_heatmap_final(
            heatmaps, dates, time_labels, cluster_names,
            output_filename=str(out_dir / out_name),
            use_percentile_clip=True,
            caption_dataset_name=dataset.title(),
        )
        plt.close(fig)
        print(f"[{dataset.upper()}] Saved -> {out_dir / out_name}")
        print(f"[{dataset.upper()}] Suggested caption:\n  {caption}")


if __name__ == "__main__":
    # Quick smoke test with synthetic data (2 clusters, 30 weeks, 48 half-hours)
    rng = np.random.default_rng(0)
    n_time, n_dates = 48, 30
    dates = list(pd.date_range("2009-07-01", periods=n_dates, freq="W"))
    time_labels = [pd.Timestamp("2000-01-01").time().replace(hour=h // 2, minute=(h % 2) * 30)
                  for h in range(n_time)]
    heatmaps = [rng.uniform(0.1, 0.4, size=(n_time, n_dates)),
                rng.uniform(0.3, 0.9, size=(n_time, n_dates)),
                rng.uniform(0.05, 0.2, size=(n_time, n_dates))]
    cluster_names = ["Cluster 0", "Cluster 1", "Cluster 2"]

    fig, caption = plot_datetime_heatmap_final(
        heatmaps, dates, time_labels, cluster_names,
        output_filename="/tmp/smoke_test_heatmap.png",
        caption_dataset_name="London",
    )
    print("Smoke test OK. Caption:", caption)
    plt.close(fig)
