"""
data_preprocessing.py
=====================
Step 1 of the Load Profiling Framework (Algorithm 2, Section 5.4.1).

All original logic preserved.  Added:
  - Column-name auto-detection  (handles trailing/leading whitespace and
    all known London CSV column-name variants across file releases)
  - Chunked CSV loading         (never loads the full 10 GB into RAM)
  - Cache helpers               (save/load processed tensor to Drive)
  - RAM monitor                 (prints available RAM at each step)
  - Max-households limit        (--max_houses CLI flag)
  - inspect_csv / check_ram     (diagnostic helpers)
"""

import numpy as np
import pandas as pd
import gc
import psutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HALF_HOURS_PER_DAY = 48
DAYS_PER_YEAR      = 365
READINGS_PER_YEAR  = DAYS_PER_YEAR * HALF_HOURS_PER_DAY   # 17 520

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ram():
    avail = psutil.virtual_memory().available / 1e9
    total = psutil.virtual_memory().total    / 1e9
    print(f"  [RAM] Available: {avail:.1f} GB / {total:.1f} GB total")
    return avail

def check_ram():
    vm    = psutil.virtual_memory()
    total = vm.total    / 1e9
    avail = vm.available/ 1e9
    used  = vm.used     / 1e9
    pct   = vm.percent
    print(f"\n[RAM Check]")
    print(f"  Total    : {total:.1f} GB")
    print(f"  Used     : {used:.1f} GB  ({pct:.0f}%)")
    print(f"  Available: {avail:.1f} GB")
    if   avail < 2.0: print("  WARNING: Less than 2 GB — risk of crash!")
    elif avail < 4.0: print("  CAUTION : Less than 4 GB available.")
    else:             print("  OK: Enough RAM available.")

def _check_n(N: int, dataset: str = ""):
    print(f"\n[Sample Check] N = {N} households loaded.")
    if   N < 100:  print(f"  CRITICAL: Only {N} — metrics will be NaN.\n"
                          "  FIX: Try --no_tariff_filter or check your CSV.")
    elif N < 500:  print(f"  WARNING: {N} is very low — recommend >= 500.")
    elif N < 1000: print(f"  CAUTION: {N} — metrics will work but may not be\n"
                          "  representative. Recommend >= 1000.")
    else:          print(f"  OK: {N} households is sufficient for clustering.")

def _find_col(variants, cols):
    """Return first matching variant (strip-safe, case-insensitive)."""
    cols_clean = [c.strip() for c in cols]
    cols_lower = {c.lower(): c for c in cols_clean}
    for v in variants:
        v_s = v.strip()
        if v_s in cols_clean:           return v_s
        if v_s.lower() in cols_lower:   return cols_lower[v_s.lower()]
    return None

# ---------------------------------------------------------------------------
# 1. London loader — chunked, memory-safe, column-name auto-detection
# ---------------------------------------------------------------------------

def _resolve_london_source_files(filepath) -> list:
    """
    [NEW] Resolves `filepath` into a list of one or more CSV files to read.

    Supports BOTH:
      - a single file path (the original raw LCL-FullData.csv)
      - a DIRECTORY containing Kaggle's halfhourly_dataset block files
        (block_0.csv, block_1.csv, ... — as they're distributed, NOT
        pre-concatenated). Mirrors how load_irish() already reads
        multiple File1.txt-File6.txt files transparently.

    Sorted for deterministic, reproducible read order across runs.
    """
    filepath = Path(filepath)
    if filepath.is_dir():
        block_files = sorted(filepath.glob("block_*.csv"))
        if not block_files:
            # Fall back to any CSVs in the folder, in case the block_*
            # naming convention doesn't match exactly
            block_files = sorted(filepath.glob("*.csv"))
        if not block_files:
            raise FileNotFoundError(
                f"{filepath} is a directory but contains no .csv files.")
        print(f"[London] Directory given — found {len(block_files)} CSV "
              f"files (e.g. {block_files[0].name} ... {block_files[-1].name})")
        return block_files
    else:
        return [filepath]


def _iter_london_chunks(source_files: list, chunk_size: int):
    """
    [NEW] Yields chunks across one or more files transparently, so every
    existing chunked-reading call site in load_london() works unchanged
    whether given a single file or Kaggle's multi-block-file folder.
    """
    for f in source_files:
        for chunk in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
            yield chunk


def load_household_tariff_info(informations_path: str) -> dict:
    """
    [NEW] Loads the Kaggle "informations_households.csv" file and returns
    a {LCLid: tariff_code} mapping.

    The Kaggle "Smart meters in London" halfhourly_dataset block files
    (LCLid, tstp, energy(kWh/hh)) do NOT include a tariff column at all —
    unlike the original raw LCL-FullData.csv, which has stdorToU inline.
    That means load_london()'s stdorToU_filter silently does nothing when
    pointed at Kaggle data (there's no "stdorToU" column to find), so
    BOTH standard-tariff AND dynamic-Time-of-Use households end up mixed
    together with no warning. This restores that filtering capability by
    cross-referencing the separate per-household metadata file the Kaggle
    version ships alongside the block files.

    Parameters
    ----------
    informations_path : path to "informations_households.csv"

    Returns
    -------
    dict {LCLid: tariff_code} e.g. {"MAC000002": "Std", "MAC000003": "ToU"}
    """
    path = Path(informations_path)
    if not path.exists():
        raise FileNotFoundError(f"Informations file not found: {informations_path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    id_col     = _find_col(["LCLid", "lclid", "ID", "id"], df.columns)
    tariff_col = _find_col(["stdorToU", "stdorTou", "tariff", "Tariff"], df.columns)

    if id_col is None or tariff_col is None:
        raise ValueError(
            f"Could not auto-detect ID/tariff columns in {informations_path}. "
            f"Found columns: {list(df.columns)}."
        )

    counts = df[tariff_col].value_counts().to_dict()
    print(f"[Tariff] Loaded {len(df)} household tariff classifications "
          f"from {path.name}")
    print(f"[Tariff] Distribution: {counts}")

    mapping = dict(zip(df[id_col].astype(str).str.strip(),
                       df[tariff_col].astype(str).str.strip()))
    return mapping


def load_london(filepath: str,
                stdorToU_filter: str = "Std",
                chunk_size: int = 500_000,
                max_houses: int = None,
                informations_households_path: str = None) -> pd.DataFrame:
    """
    Load the Low Carbon London smart-meter CSV.

    Handles ALL known column-name variants including trailing/leading
    whitespace (e.g. 'KWH/hh (per half hour) ' with trailing space).
    Reads the CSV in chunks so it never loads the full 10 GB into RAM.

    Parameters
    ----------
    filepath        : path to LCL-FullData.csv (or a concatenated Kaggle
                      halfhourly_dataset — see informations_households_path
                      below if using the Kaggle version)
    stdorToU_filter : 'Std', 'ToU', or None (None = use all households)
    chunk_size      : rows per chunk (reduce to 200_000 if crashing)
    max_houses      : max households to load (None = all)
                      Free Colab → 1000-2000 | Colab Pro → None
    informations_households_path : [NEW] path to Kaggle's
                      "informations_households.csv". The Kaggle
                      halfhourly_dataset block files have NO tariff
                      column inline (unlike the original LCL-FullData.csv)
                      — without this, stdorToU_filter silently does
                      nothing on Kaggle data, mixing Std and ToU
                      households together with no warning. Pass this to
                      restore proper filtering via cross-reference. Not
                      needed if filepath already has a native stdorToU
                      column (i.e. the original raw file).

    Returns
    -------
    wide : DataFrame  (timestamps × households)
    """
    filepath = Path(filepath)
    source_files = _resolve_london_source_files(filepath)
    print(f"\n[London] Reading {filepath.name if filepath.is_file() else filepath} ...")
    _ram()

    # ── Detect column names from header only (strip whitespace) ──────────
    # Uses the FIRST file when given a directory — every Kaggle block file
    # shares the same column structure, so this is representative.
    header_df = pd.read_csv(source_files[0], nrows=0)
    header_df.columns = [c.strip() for c in header_df.columns]
    actual_cols = list(header_df.columns)
    print(f"[London] Columns found (after strip): {actual_cols}")

    ENERGY_VARIANTS  = ["KWH/hh (per half hour)", "energy(kWh/hh)",
                        "kWh/hh", "Consumption", "energy"]
    TIME_VARIANTS    = ["DateTime", "tstp", "timestamp", "Datetime"]
    ID_VARIANTS      = ["LCLid", "lclid", "ID", "id", "HouseholdID"]
    TARIFF_VARIANTS  = ["stdorToU", "stdorTou", "tariff", "Tariff"]

    energy_col = _find_col(ENERGY_VARIANTS, actual_cols)
    time_col   = _find_col(TIME_VARIANTS,   actual_cols)
    id_col     = _find_col(ID_VARIANTS,     actual_cols)
    tariff_col = _find_col(TARIFF_VARIANTS, actual_cols)

    print(f"[London] Detected → ID:{id_col}  Time:{time_col}  "
          f"Energy:{energy_col}  Tariff:{tariff_col}")

    if energy_col is None or time_col is None or id_col is None:
        raise ValueError(
            f"Could not detect required columns.\n"
            f"Columns present: {actual_cols}\n"
            f"Energy expected: {ENERGY_VARIANTS}"
        )

    # [FIX] Previously, if no tariff column was found (e.g. the Kaggle
    # halfhourly_dataset format, which has no inline tariff column),
    # stdorToU_filter would silently do nothing — Std and ToU households
    # got mixed together with zero warning. Now: either load an external
    # mapping to restore proper filtering, or warn LOUDLY that filtering
    # is being skipped, so this is a known, visible tradeoff rather than
    # a silent one.
    external_tariff_map = None
    if tariff_col is None and stdorToU_filter:
        if informations_households_path:
            print(f"[London] No inline tariff column found — cross-"
                  f"referencing {informations_households_path} instead ...")
            external_tariff_map = load_household_tariff_info(
                informations_households_path)
        else:
            print(f"\n{'!'*70}")
            print(f"[London] WARNING: stdorToU_filter='{stdorToU_filter}' was "
                  f"requested, but no tariff column was found in this file "
                  f"(columns: {actual_cols}). This is EXPECTED if you're "
                  f"using the Kaggle 'halfhourly_dataset' format, which has "
                  f"no inline tariff column. FILTERING WILL BE SKIPPED — "
                  f"both Std and ToU households will be included together. "
                  f"To restore proper filtering, pass "
                  f"informations_households_path pointing at Kaggle's "
                  f"'informations_households.csv'.")
            print(f"{'!'*70}\n")

    # ── Inner helpers (no usecols — avoids whitespace mismatch) ──────────
    def _clean(chunk):
        """Strip, rename, type-convert one chunk. Returns 3-col DataFrame."""
        chunk.columns = [c.strip() for c in chunk.columns]
        keep = {id_col, time_col, energy_col}
        if tariff_col and tariff_col in chunk.columns:
            keep.add(tariff_col)
        chunk = chunk[[c for c in chunk.columns if c in keep]].copy()
        rename = {id_col: "LCLid", time_col: "DateTime", energy_col: "kwh"}
        if tariff_col and tariff_col in chunk.columns:
            rename[tariff_col] = "stdorToU"
        chunk = chunk.rename(columns=rename)
        if stdorToU_filter and "stdorToU" in chunk.columns:
            chunk = chunk[chunk["stdorToU"] == stdorToU_filter]
        elif stdorToU_filter and external_tariff_map is not None:
            # [FIX] No inline tariff column (Kaggle format) — filter using
            # the cross-referenced external mapping instead.
            ids_clean = chunk["LCLid"].astype(str).str.strip()
            keep_mask = ids_clean.map(
                lambda lclid: external_tariff_map.get(lclid) == stdorToU_filter
            )
            chunk = chunk[keep_mask.values]
        chunk["DateTime"] = pd.to_datetime(chunk["DateTime"], errors="coerce")
        chunk["kwh"]      = pd.to_numeric(chunk["kwh"], errors="coerce")
        chunk = chunk.dropna(subset=["DateTime"])
        return chunk[["LCLid", "DateTime", "kwh"]]

    def _clean_notariff(chunk):
        """Same as _clean but skips tariff filter (used in fallback)."""
        chunk.columns = [c.strip() for c in chunk.columns]
        keep = {id_col, time_col, energy_col}
        chunk = chunk[[c for c in chunk.columns if c in keep]].copy()
        rename = {id_col: "LCLid", time_col: "DateTime", energy_col: "kwh"}
        chunk = chunk.rename(columns=rename)
        chunk["DateTime"] = pd.to_datetime(chunk["DateTime"], errors="coerce")
        chunk["kwh"]      = pd.to_numeric(chunk["kwh"], errors="coerce")
        chunk = chunk.dropna(subset=["DateTime"])
        return chunk[["LCLid", "DateTime", "kwh"]]

    # ── Pass 1: collect household IDs (if max_houses set) ─────────────────
    selected_ids = None
    if max_houses is not None:
        print(f"[London] Pass 1: collecting up to {max_houses} household IDs ...")
        house_ids = set()
        for chunk in _iter_london_chunks(source_files, chunk_size):
            c = _clean(chunk)
            house_ids.update(c["LCLid"].unique())
            if len(house_ids) >= max_houses:
                break
        selected_ids = sorted(house_ids)[:max_houses]
        print(f"[London] Selected {len(selected_ids)} households.")
        _ram()

    # ── Pass 2: read all data ─────────────────────────────────────────────
    print(f"[London] Pass 2: reading data (chunk_size={chunk_size:,}) ...")
    frames, n = [], 0
    for chunk in _iter_london_chunks(source_files, chunk_size):
        n += 1
        c  = _clean(chunk)
        if selected_ids is not None:
            c = c[c["LCLid"].isin(selected_ids)]
        if not c.empty:
            frames.append(c)
        if n % 20 == 0:
            print(f"  Chunk {n} ...")
            _ram()

    print(f"[London] {n} chunks read. Concatenating ...")
    df = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()

    # ── Auto-fallback: retry without tariff filter if too few households ──
    n_houses = df["LCLid"].nunique()
    print(f"[London] Unique households: {n_houses}")

    if n_houses < 200 and stdorToU_filter:
        print(f"\n  WARNING: Only {n_houses} households match "
              f"tariff='{stdorToU_filter}'.")
        print("  Retrying WITHOUT tariff filter ...")
        frames2, n2 = [], 0
        for chunk in _iter_london_chunks(source_files, chunk_size):
            n2 += 1
            c = _clean_notariff(chunk)
            if selected_ids is not None:
                c = c[c["LCLid"].isin(selected_ids)]
            if not c.empty:
                frames2.append(c)
        df = pd.concat(frames2, ignore_index=True)
        del frames2; gc.collect()
        n_houses = df["LCLid"].nunique()
        print(f"[London] After retry: {n_houses} households")

    _ram()

    # ── Pivot to wide format ──────────────────────────────────────────────
    print("[London] Pivoting to wide format ...")
    wide = df.pivot_table(index="DateTime", columns="LCLid",
                          values="kwh", aggfunc="first")
    wide.sort_index(inplace=True)
    del df; gc.collect()
    print(f"[London] Wide shape: {wide.shape}  (timestamps × households)")
    _ram()
    return wide


# ---------------------------------------------------------------------------
# 2. Irish loader
# ---------------------------------------------------------------------------

def load_irish(data_dir: str) -> pd.DataFrame:
    """
    Load the Irish CER smart-meter dataset.

    Each text file has three columns (no header):
        MeterID | DDDTT  (day-code + time-code, 5 digits) | kWh

    Day 1  = 1 Jan 2009
    Time codes 01-48 = 30-min slots

    Returns a wide DataFrame: rows = timestamps, columns = meter IDs.
    """
    data_dir = Path(data_dir)
    chunks   = []

    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        txt_files = [f for f in sorted(data_dir.glob("*")) if f.is_file()]

    print(f"[Irish] Found {len(txt_files)} files in {data_dir}")
    _ram()

    for fpath in txt_files:
        print(f"[Irish] Reading {fpath.name} ...")
        try:
            chunk = pd.read_csv(fpath, sep=" ", header=None,
                                names=["meter_id", "dddtt", "kwh"])
            chunks.append(chunk)
        except Exception as e:
            print(f"  Warning: could not read {fpath.name}: {e}")

    if not chunks:
        raise ValueError(f"No Irish data files found in {data_dir}")

    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    # Parse the 5-digit DDDTT field
    df["day_code"]  = df["dddtt"].astype(str).str.zfill(5).str[:3].astype(int)
    df["time_code"] = df["dddtt"].astype(str).str.zfill(5).str[3:].astype(int)

    # Reconstruct actual timestamp
    base = pd.Timestamp("2009-01-01")
    df["DateTime"] = (
        base
        + pd.to_timedelta(df["day_code"] - 1, unit="D")
        + pd.to_timedelta((df["time_code"] - 1) * 30, unit="min")
    )

    wide = df.pivot_table(index="DateTime", columns="meter_id",
                          values="kwh", aggfunc="first")
    wide.sort_index(inplace=True)
    del df; gc.collect()
    print(f"[Irish] Wide shape: {wide.shape}  (timestamps × households)")
    _ram()
    return wide


# ---------------------------------------------------------------------------
# 3. Missing-value imputation  (Eq. 26)
# ---------------------------------------------------------------------------

def impute_missing(wide: pd.DataFrame,
                   missing_threshold: float = 1.0) -> pd.DataFrame:
    """
    Fast vectorised imputation of NaN values following Eq. 26.

    Strategy (applied in order, all operations are vectorised):
      1. Same slot +1 year  (shift by READINGS_PER_YEAR rows upward)
      2. Same slot -1 year  (shift by READINGS_PER_YEAR rows downward)
      3. Monthly column mean (per-month average across all timestamps)

    Drops households with > missing_threshold fraction of NaNs.
    Default threshold = 1.0 (keep ALL households, matching the paper).

    Speed: ~50x faster than the original row-by-row Python loop.
    London (4438 hh x 39727 ts): ~30 seconds instead of ~30 minutes.
    """
    import time
    t0 = time.time()

    step = READINGS_PER_YEAR    # 17520 half-hourly slots per year

    # ── Drop households above the missing threshold ───────────────────────
    missing_frac = wide.isnull().mean(axis=0)
    keep         = missing_frac[missing_frac <= missing_threshold].index
    dropped      = len(wide.columns) - len(keep)
    if dropped:
        print(f"[Impute] Dropped {dropped} households "
              f"(>{missing_threshold*100:.0f}% missing)")
    else:
        print(f"[Impute] Keeping all {len(keep)} households "
              f"(threshold={missing_threshold*100:.0f}%)")
    wide = wide[keep].copy()

    arr       = wide.values.astype(float)   # (T, N)
    T, N      = arr.shape
    nan_mask  = np.isnan(arr)               # True where value is missing
    print(f"[Impute] Total NaNs before imputation: {nan_mask.sum():,} "
          f"({nan_mask.mean()*100:.1f}% of all values)")

    # ── Step 1: fill from +1 year (vectorised row shift) ─────────────────
    # For each row i, take the value from row i+step if available
    if T > step:
        src_plus  = np.full_like(arr, np.nan)
        src_plus[:T-step, :] = arr[step:, :]       # shift upward by step
        fill_mask = nan_mask & ~np.isnan(src_plus)
        arr[fill_mask]     = src_plus[fill_mask]
        nan_mask           = np.isnan(arr)
        print(f"[Impute] After +1 year fill: {nan_mask.sum():,} NaNs remaining")

    # ── Step 2: fill from -1 year (vectorised row shift) ─────────────────
    if T > step:
        src_minus = np.full_like(arr, np.nan)
        src_minus[step:, :] = arr[:T-step, :]      # shift downward by step
        fill_mask = nan_mask & ~np.isnan(src_minus)
        arr[fill_mask]    = src_minus[fill_mask]
        nan_mask          = np.isnan(arr)
        print(f"[Impute] After -1 year fill: {nan_mask.sum():,} NaNs remaining")
        del src_plus, src_minus
        gc.collect()

    # ── Step 3: monthly column mean (vectorised per month) ────────────────
    if nan_mask.any():
        months = wide.index.month.values          # (T,) integer month 1-12
        for m in range(1, 13):
            month_rows = (months == m)            # boolean mask of rows in month m
            if not month_rows.any():
                continue
            # Compute mean per column for this month, ignoring NaN
            col_means = np.nanmean(arr[month_rows, :], axis=0)  # (N,)
            # Rows in this month that are still NaN
            nan_in_month = nan_mask & month_rows[:, None]
            if nan_in_month.any():
                # Broadcast col_means across the NaN positions
                arr[nan_in_month] = np.broadcast_to(
                    col_means, arr.shape
                )[nan_in_month]
        nan_mask = np.isnan(arr)
        print(f"[Impute] After monthly mean fill: {nan_mask.sum():,} NaNs remaining")

    # ── Step 4: global mean fallback (should rarely trigger) ─────────────
    if nan_mask.any():
        global_means = np.nanmean(arr, axis=0)
        for j in range(N):
            col_nan = nan_mask[:, j]
            if col_nan.any():
                arr[col_nan, j] = (global_means[j]
                                   if not np.isnan(global_means[j]) else 0.0)
        print(f"[Impute] After global mean fill: {np.isnan(arr).sum():,} NaNs remaining")

    wide_imputed  = pd.DataFrame(arr, index=wide.index, columns=wide.columns)
    remaining_nan = wide_imputed.isnull().sum().sum()
    elapsed       = time.time() - t0
    print(f"[Impute] Done in {elapsed:.1f}s — "
          f"remaining NaNs: {remaining_nan}")
    gc.collect()
    return wide_imputed


# ---------------------------------------------------------------------------
# 4. Min-Max normalisation  (Eq. 27)
# ---------------------------------------------------------------------------

def normalize_minmax(wide: pd.DataFrame) -> np.ndarray:
    """
    Per-household Min-Max normalisation.

    Eq. 27: x_ji = (x_ji - min(x_i)) / (max(x_i) - min(x_i))

    Returns numpy array (N_households, T_timestamps).
    """
    arr     = wide.values.astype(float)        # (T, N)
    col_min = arr.min(axis=0, keepdims=True)
    col_max = arr.max(axis=0, keepdims=True)
    denom   = col_max - col_min
    denom[denom == 0] = 1.0
    arr_norm  = (arr - col_min) / denom
    data_norm = arr_norm.T                     # (N, T)
    print(f"[Normalise] Shape (households × timestamps): {data_norm.shape}")
    gc.collect()
    return data_norm


# ---------------------------------------------------------------------------
# 5. Reshape to 5-D ConvLSTM tensor
# ---------------------------------------------------------------------------

def reshape_for_convlstm(data:       np.ndarray,
                          n_steps:   int = 7,
                          n_length:  int = 48,
                          n_features:int = 1) -> np.ndarray:
    """
    Reshape (N, T) → (N, n_steps, 1, n_length, n_features).

    Uses the most recent n_steps*n_length timestamps per household.
    """
    N, T   = data.shape
    window = n_steps * n_length

    if T < window:
        raise ValueError(
            f"Time axis ({T}) shorter than required window "
            f"({n_steps}×{n_length}={window}). "
            "Increase n_length or reduce n_steps."
        )

    data_trimmed = data[:, -window:]
    X = data_trimmed.reshape(N, n_steps, 1, n_length, n_features)
    print(f"[Reshape] 5-D tensor: {X.shape} "
          f"(samples, n_steps, height, n_length, channels)")
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Cache helpers
# ---------------------------------------------------------------------------

def save_preprocessed(X: np.ndarray, wide: pd.DataFrame,
                       save_dir: str, dataset: str = "london"):
    """Save tensor and wide DataFrame so next run loads instantly."""
    save_dir  = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    x_path    = save_dir / f"{dataset}_X.npy"
    wide_path = save_dir / f"{dataset}_wide.parquet"
    np.save(str(x_path), X)
    wide.to_parquet(str(wide_path))
    print(f"[Cache] Saved tensor : {x_path}  "
          f"({x_path.stat().st_size/1e6:.0f} MB)")
    print(f"[Cache] Saved wide   : {wide_path}  "
          f"({wide_path.stat().st_size/1e6:.0f} MB)")


def load_preprocessed(save_dir: str, dataset: str = "london"):
    """Load cached tensor and wide DataFrame. Returns (None, None) if missing."""
    save_dir  = Path(save_dir)
    x_path    = save_dir / f"{dataset}_X.npy"
    wide_path = save_dir / f"{dataset}_wide.parquet"
    if not x_path.exists() or not wide_path.exists():
        return None, None
    print(f"[Cache] Loading {dataset} from cache ...")
    X    = np.load(str(x_path))
    wide = pd.read_parquet(str(wide_path))
    print(f"[Cache] Tensor: {X.shape}   Wide: {wide.shape}")
    _ram()
    return X, wide


def clear_cache(cache_dir: str, dataset: str = None):
    """
    Delete cached files so preprocessing re-reads from the raw CSV/files.

    Parameters
    ----------
    cache_dir : path to cache folder
    dataset   : 'london', 'irish', or None (clears both)

    Called automatically when force_reprocess=True is passed to
    preprocess_london() or preprocess_irish().
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        print(f"[Cache] Cache folder does not exist: {cache_dir}")
        return

    datasets = [dataset] if dataset else ["london", "irish"]
    deleted  = []

    for ds in datasets:
        for suffix in ["_X.npy", "_wide.parquet"]:
            p = cache_dir / f"{ds}{suffix}"
            if p.exists():
                p.unlink()
                deleted.append(p.name)

    if deleted:
        print(f"[Cache] Cleared: {deleted}")
    else:
        print(f"[Cache] Nothing to clear for {datasets}.")


# ---------------------------------------------------------------------------
# 7. Full preprocessing pipelines
# ---------------------------------------------------------------------------

def preprocess_london(filepath:          str,
                      stdorToU_filter:   str   = "Std",
                      missing_threshold: float = 1.0,
                      n_steps:           int   = 7,
                      n_length:          int   = 48,
                      chunk_size:        int   = 500_000,
                      max_houses:        int   = None,
                      cache_dir:         str   = None,
                      force_reprocess:   bool  = False,
                      informations_households_path: str = None):
    """
    Memory-safe end-to-end preprocessing for the London dataset.

    Parameters
    ----------
    force_reprocess : if True, deletes existing cache and re-reads the CSV.
                      Use this whenever you want fresh data (e.g. after
                      changing max_houses or fixing a bug).
                      Default is False so repeated runs reuse the cache.
    informations_households_path : [NEW] path to Kaggle's "informations_
                      households.csv" — only needed if filepath is the
                      Kaggle halfhourly_dataset (concatenated blocks),
                      which has no inline tariff column. Not needed for
                      the original raw LCL-FullData.csv.

    Returns (X, wide):
        X    : numpy array (N, n_steps, 1, n_length, 1)  — model input
        wide : DataFrame   (timestamps × households)       — for load profiling
    """
    print("\n" + "="*60)
    print("LONDON PREPROCESSING — MEMORY SAFE")
    if max_houses:
        print(f"Using up to {max_houses} households "
              f"(set max_houses=None for all)")
    print("="*60)
    _ram()

    # Delete old cache if force_reprocess is set
    if force_reprocess and cache_dir:
        print("[Cache] force_reprocess=True — clearing old London cache ...")
        clear_cache(cache_dir, "london")

    if cache_dir and not force_reprocess:
        X, wide = load_preprocessed(cache_dir, "london")
        if X is not None:
            print("[Cache] Using cached data — skipping CSV read.")
            print(f"[Cache] Households in cache: {X.shape[0]}")
            print("        To re-read the CSV, pass force_reprocess=True")
            _check_n(X.shape[0], "london")
            return X, wide

    wide      = load_london(filepath, stdorToU_filter, chunk_size, max_houses,
                            informations_households_path=informations_households_path)
    wide      = impute_missing(wide, missing_threshold)
    _check_n(wide.shape[1], "london")
    data_norm = normalize_minmax(wide)
    X         = reshape_for_convlstm(data_norm, n_steps, n_length)

    if cache_dir:
        save_preprocessed(X, wide, cache_dir, "london")

    return X, wide


def load_residential_allocation(allocation_path: str) -> set:
    """
    [NEW] Loads the CER "SME and Residential allocations" file and returns
    the set of meter IDs classified as RESIDENTIAL (Code == 1).

    The official CER Smart Metering Trial dataset mixes THREE participant
    types in the same six raw reading files (File1.txt-File6.txt) with no
    type label in those files themselves: Residential (Code 1), SME/small
    business (Code 2), and Other (Code 3). Without cross-referencing this
    separate allocations file, every downstream "household" in the
    pipeline may actually include a meaningful number of commercial
    meters — structurally different consumption patterns (weekday-
    dominant, near-zero weekends, different magnitude) that can distort
    clustering aimed specifically at residential segmentation.

    Column names in the distributed file are not perfectly standardised
    across releases, so this auto-detects the ID and Code columns by
    common name patterns rather than hardcoding exact strings.

    Parameters
    ----------
    allocation_path : path to "SME and Residential allocations.xlsx" or
                      ".csv" (as distributed by CER/ISSDA)

    Returns
    -------
    set of meter IDs (as strings, matching load_irish()'s meter_id dtype)
    classified as Code == 1 (Residential).
    """
    path = Path(allocation_path)
    if not path.exists():
        raise FileNotFoundError(f"Allocation file not found: {allocation_path}")

    if path.suffix.lower() in (".xlsx", ".xls"):
        # [FIX] pandas sometimes fails to auto-detect the Excel engine even
        # when openpyxl IS installed ("Excel file format cannot be
        # determined"). Try explicit engines first; if all fail AND a same-
        # named .csv sibling exists (as it does for the standard CER
        # release, which ships both formats), fall back to that instead of
        # erroring out — same data, no engine dependency at all.
        df = None
        last_error = None
        for engine in ("openpyxl", "xlrd", None):
            try:
                df = (pd.read_excel(path, engine=engine) if engine
                     else pd.read_excel(path))
                break
            except Exception as e:
                last_error = e
        if df is None:
            csv_sibling = path.with_suffix(".csv")
            if csv_sibling.exists():
                print(f"[Allocation] Could not read {path.name} as Excel "
                      f"({last_error}) — found a .csv version at "
                      f"{csv_sibling.name}, using that instead.")
                df = pd.read_csv(csv_sibling)
            else:
                raise ValueError(
                    f"Could not read {allocation_path} as Excel "
                    f"({last_error}), and no .csv sibling file was found "
                    f"to fall back to. Try installing openpyxl "
                    f"(pip install openpyxl --break-system-packages) or "
                    f"point --irish_allocation_path at the .csv version "
                    f"directly if you have one."
                )
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]

    id_col = _find_col(["ID", "Meter ID", "MeterID", "meter_id"], df.columns)
    code_col = _find_col(["Code", "code", "Sample", "sample", "Allocation"],
                         df.columns)

    if id_col is None or code_col is None:
        raise ValueError(
            f"Could not auto-detect ID/Code columns in {allocation_path}. "
            f"Found columns: {list(df.columns)}. Expected something like "
            f"'ID' and 'Code' (Code: 1=Residential, 2=SME, 3=Other)."
        )

    counts = df[code_col].value_counts().to_dict()
    print(f"[Allocation] Loaded {len(df)} meter classifications from "
          f"{path.name}")
    print(f"[Allocation] Code distribution: {counts} "
          f"(1=Residential, 2=SME, 3=Other, per CER documentation)")

    residential_ids = set(
        df.loc[df[code_col] == 1, id_col].astype(str).str.strip()
    )
    print(f"[Allocation] Residential (Code==1) meter IDs: {len(residential_ids)}")
    return residential_ids


def filter_to_residential(wide: pd.DataFrame, allocation_path: str) -> pd.DataFrame:
    """
    [NEW] Restricts `wide`'s columns (meter IDs) to residential-only,
    using load_residential_allocation(). Prints exactly how many
    households were dropped as SME/Other/unclassified so the filtering
    is transparent, not silent.
    """
    residential_ids = load_residential_allocation(allocation_path)
    wide_ids = set(str(c).strip() for c in wide.columns)

    keep_ids = wide_ids & residential_ids
    dropped_non_residential = wide_ids - residential_ids

    print(f"[Allocation] {len(wide.columns)} meters in the raw data files")
    print(f"[Allocation] {len(keep_ids)} match a Residential (Code==1) ID")
    print(f"[Allocation] {len(dropped_non_residential)} dropped "
          f"(SME/Other/not found in allocation file)")

    if len(keep_ids) == 0:
        raise ValueError(
            "Zero meter IDs matched after residential filtering — check "
            "that the allocation file's ID column format matches the raw "
            "data files' meter_id format (e.g. both as plain integers, "
            "not one zero-padded and the other not)."
        )

    # Preserve original column order/dtype, just restricted to residential IDs
    keep_cols = [c for c in wide.columns if str(c).strip() in keep_ids]
    return wide[keep_cols]


def preprocess_irish(data_dir:          str,
                     missing_threshold: float = 1.0,
                     n_steps:           int   = 7,
                     n_length:          int   = 48,
                     cache_dir:         str   = None,
                     force_reprocess:   bool  = False,
                     allocation_path:   str   = None):
    """
    End-to-end preprocessing for the Irish dataset.

    Parameters
    ----------
    force_reprocess : if True, deletes existing cache and re-reads the files.
    allocation_path : [NEW] optional path to the CER "SME and Residential
                      allocations" file. If given, restricts the dataset
                      to Code==1 (Residential) meter IDs only, excluding
                      SME/commercial meters that are otherwise mixed in
                      with no type label in the raw six data files. If
                      None (default), no filtering happens — matches the
                      original all-meters behaviour.

    Returns (X, wide).
    """
    print("\n" + "="*60)
    print("IRISH PREPROCESSING")
    print("="*60)
    _ram()

    # Delete old cache if force_reprocess is set
    if force_reprocess and cache_dir:
        print("[Cache] force_reprocess=True — clearing old Irish cache ...")
        clear_cache(cache_dir, "irish")

    if cache_dir and not force_reprocess:
        X, wide = load_preprocessed(cache_dir, "irish")
        if X is not None:
            print("[Cache] Using cached data — skipping file read.")
            print(f"[Cache] Households in cache: {X.shape[0]}")
            print("        To re-read the files, pass force_reprocess=True")
            _check_n(X.shape[0], "irish")
            return X, wide

    wide      = load_irish(data_dir)
    if allocation_path:
        print(f"\n[Allocation] Filtering to residential-only households "
              f"using {allocation_path} ...")
        wide = filter_to_residential(wide, allocation_path)
    wide      = impute_missing(wide, missing_threshold)
    _check_n(wide.shape[1], "irish")
    data_norm = normalize_minmax(wide)
    X         = reshape_for_convlstm(data_norm, n_steps, n_length)

    if cache_dir:
        save_preprocessed(X, wide, cache_dir, "irish")

    return X, wide


# ---------------------------------------------------------------------------
# 8. Diagnostics
# ---------------------------------------------------------------------------

def inspect_csv(filepath: str, n_rows: int = 3):
    """Print column names and first n_rows. Call this if you get column errors."""
    filepath = Path(filepath)
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath.name}")
    print(f"{'='*60}")
    header = pd.read_csv(filepath, nrows=0)
    print(f"\nColumn names ({len(header.columns)} total):")
    for i, col in enumerate(header.columns):
        print(f"  [{i}]  repr={repr(col)}  stripped={repr(col.strip())}")
    sample = pd.read_csv(filepath, nrows=n_rows)
    print(f"\nFirst {n_rows} rows:")
    print(sample.to_string())
    print(f"\nFile size : {filepath.stat().st_size / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    return list(header.columns)


if __name__ == "__main__":
    print("data_preprocessing.py — loaded successfully.")
    check_ram()
