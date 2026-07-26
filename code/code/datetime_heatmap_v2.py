"""
Date x Time-of-Day heatmap, layout v2 — no overall title (goes in the
LaTeX caption instead), single shared x-axis label, exact typography spec.

This is your target implementation, used essentially as given. The only
change: time_labels here comes from compute_cluster_heatmaps() as
datetime.time objects (not "00:00"-style strings), so the function accepts
either — datetime.time objects are auto-formatted to "%H:%M" strings
internally before the lookups your code relies on
(`time_labels.index(target)`). Nothing about the data, the heatmap
calculation, cluster assignments, color values, date range, time
resolution, or shared color-scale logic is touched — this file is
presentation-layer only, same as v1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_date_time_heatmaps(
    heatmaps,
    dates,
    time_labels,
    output_path,
    cluster_names=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
):
    """
    Create publication-quality date x time-of-day heatmaps.

    Parameters
    ----------
    heatmaps : list of np.ndarray
        Three arrays, each with shape (n_time_slots, n_dates).
    dates : array-like
        Dates corresponding to heatmap columns.
    time_labels : list of str OR list of datetime.time
        Time labels corresponding to heatmap rows. datetime.time objects
        (e.g. straight from compute_cluster_heatmaps()) are auto-converted
        to "%H:%M" strings; plain strings are used as-is.
    output_path : str
        Output PNG path.
    cluster_names : list of str, optional
        Panel titles.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float, optional
        Shared color limits.
    """

    if len(heatmaps) != 3:
        raise ValueError("Exactly three heatmaps are required.")

    if cluster_names is None:
        cluster_names = ["Cluster 0", "Cluster 1", "Cluster 2"]

    if len(cluster_names) != 3:
        raise ValueError("Exactly three cluster names are required.")

    dates = pd.to_datetime(dates)

    # Accept datetime.time objects (compute_cluster_heatmaps' native output)
    # or plain strings, transparently.
    time_labels = [
        t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)
        for t in time_labels
    ]

    matrices = [np.asarray(h, dtype=float) for h in heatmaps]

    for i, matrix in enumerate(matrices):
        if matrix.ndim != 2:
            raise ValueError(f"Heatmap {i} must be a 2D array.")

        if matrix.shape[1] != len(dates):
            raise ValueError(
                f"Heatmap {i} has {matrix.shape[1]} columns, "
                f"but {len(dates)} dates were provided."
            )

        if matrix.shape[0] != len(time_labels):
            raise ValueError(
                f"Heatmap {i} has {matrix.shape[0]} rows, "
                f"but {len(time_labels)} time labels were provided."
            )

    finite_values = np.concatenate(
        [m[np.isfinite(m)] for m in matrices]
    )

    if finite_values.size == 0:
        raise ValueError("The heatmaps contain no finite values.")

    if vmin is None:
        vmin = np.nanmin(finite_values)

    if vmax is None:
        vmax = np.nanmax(finite_values)

    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin.")

    fig = plt.figure(
        figsize=(17, 5.8),
        facecolor="white",
    )

    grid = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=[1, 1, 1, 0.05],
        left=0.065,
        right=0.94,
        bottom=0.20,
        top=0.90,
        wspace=0.18,
    )

    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    colorbar_axis = fig.add_subplot(grid[0, 3])

    date_numbers = mdates.date2num(dates)

    date_start = date_numbers.min()
    date_end = date_numbers.max()

    images = []

    for index, (ax, matrix, cluster_name) in enumerate(
        zip(axes, matrices, cluster_names)
    ):
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[
                date_start,
                date_end,
                len(time_labels) - 0.5,
                -0.5,
            ],
        )

        images.append(image)

        ax.set_title(
            cluster_name,
            fontsize=16,
            fontweight="bold",
            pad=8,
        )

        # Exactly five approximately even date ticks
        tick_positions = np.linspace(
            date_start,
            date_end,
            5,
        )

        ax.set_xticks(tick_positions)
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b %Y")
        )

        plt.setp(
            ax.get_xticklabels(),
            rotation=30,
            ha="right",
            rotation_mode="anchor",
            fontsize=11,
        )

        selected_times = ["00:00", "04:00", "08:00",
                          "12:00", "16:00", "20:00"]

        selected_positions = []

        for target in selected_times:
            if target not in time_labels:
                raise ValueError(
                    f"Required time label {target} was not found."
                )

            selected_positions.append(
                time_labels.index(target)
            )

        if index == 0:
            ax.set_yticks(selected_positions)
            ax.set_yticklabels(
                selected_times,
                fontsize=11,
            )

            ax.set_ylabel(
                "Time of day",
                fontsize=13,
            )
        else:
            ax.set_yticks(selected_positions)
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=11,
        )

        # Remove individual x-axis labels
        ax.set_xlabel("")

    colorbar = fig.colorbar(
        images[-1],
        cax=colorbar_axis,
    )

    colorbar.set_label(
        "Median electricity consumption (kWh/30 min)",
        fontsize=13,
        labelpad=12,
    )

    colorbar.ax.tick_params(
        labelsize=11,
    )

    # One shared x-axis label
    fig.supxlabel(
        "Date",
        fontsize=13,
        y=0.06,
    )

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# ===========================================================================
# Example calls — London and Irish, reusing the already-fast
# compute_cluster_heatmaps() (no melt, ~seconds not minutes) from
# datetime_heatmap_final.py, feeding straight into the v2 plotting layout.
# ===========================================================================

def generate_both_datetime_heatmaps_v2(results_base_dir, cache_dir, output_dir="."):
    """
    Same artifact-loading approach as generate_both_datetime_heatmaps() in
    datetime_heatmap_final.py (no retraining, no re-clustering) — just
    wired to the new plot_date_time_heatmaps() v2 layout and v2 filenames.
    """
    from pathlib import Path
    from generate_figures import load_run_artifacts
    from comprehensive_dashboard import _households_by_cluster
    from datetime_heatmap_final import compute_cluster_heatmaps

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, out_name in [("london", "London_DateTimeHeatmap_Final_v2.png"),
                              ("irish",  "Irish_DateTimeHeatmap_Final_v2.png")]:
        print(f"\n[{dataset.upper()}] Loading saved artifacts …")
        art = load_run_artifacts(
            results_dir=f"{results_base_dir}/{dataset}",
            cache_dir=cache_dir, dataset=dataset)

        wide, labels, K = art["wide"], art["labels"], art["K"]
        if K != 3:
            raise ValueError(
                f"{dataset}: plot_date_time_heatmaps() requires exactly "
                f"K=3 clusters, but this run's optimal K={K}."
            )
        household_ids = list(wide.columns)
        by_cluster = _households_by_cluster(household_ids, labels, K)

        heatmaps, dates, time_labels = compute_cluster_heatmaps(wide, by_cluster, K)
        cluster_names = [f"Cluster {k}" for k in range(K)]

        plot_date_time_heatmaps(
            heatmaps=heatmaps,
            dates=dates,
            time_labels=time_labels,
            output_path=str(out_dir / out_name),
            cluster_names=cluster_names,
            cmap="viridis",
            vmin=None,   # shared min/max computed across all 3 panels internally
            vmax=None,
        )
        print(f"[{dataset.upper()}] Saved -> {out_dir / out_name}")


if __name__ == "__main__":
    # Quick smoke test with synthetic data (3 clusters, half-hourly time labels)
    rng = np.random.default_rng(0)
    n_time, n_dates = 48, 60
    dates = list(pd.date_range("2009-07-01", periods=n_dates, freq="W"))
    time_labels = [
        pd.Timestamp("2000-01-01").time().replace(hour=h // 2, minute=(h % 2) * 30)
        for h in range(n_time)
    ]
    heatmaps = [rng.uniform(0.1, 0.4, size=(n_time, n_dates)),
                rng.uniform(0.3, 0.9, size=(n_time, n_dates)),
                rng.uniform(0.05, 0.2, size=(n_time, n_dates))]

    plot_date_time_heatmaps(
        heatmaps=heatmaps,
        dates=dates,
        time_labels=time_labels,
        output_path="/tmp/smoke_test_heatmap_v2.png",
        cluster_names=["Cluster 0", "Cluster 1", "Cluster 2"],
    )
    print("Smoke test OK -> /tmp/smoke_test_heatmap_v2.png")
