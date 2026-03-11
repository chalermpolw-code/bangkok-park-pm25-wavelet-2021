#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Year-replication QC scan for outside stations (2021/2022/2023).

Purpose
-------
This is a station-level pre-QC script for wavelet replication.
It does NOT decide final wavelet usability by itself.
Final usability must still be checked at the park–outside PAIR level.

What it does
------------
1) Reads all outside station CSVs from analysis/data_processed/outside_hourly
2) Parses/normalizes time to Asia/Bangkok
3) Keeps only whole-hour rows
4) Drops duplicate timestamps
5) Flags obviously broken early timestamps (e.g., 1970 issue)
6) For each requested year:
   - clips to that year window
   - reindexes to full hourly grid
   - converts negative PM2.5 to NaN
   - computes coverage / fragmentation metrics
   - writes year-specific cleaned hourly file
7) Writes a summary CSV for all station-year combinations

Recommended next step after this
--------------------------------
Run a PAIR-level overlap QC script before any 2022/2023 wavelet analysis.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

TZ = "Asia/Bangkok"

DEFAULT_IN_DIR = "analysis/data_processed/outside_hourly"
DEFAULT_OUT_DIR = "analysis/data_processed/outside_hourly_clean_yearscan"
DEFAULT_OUT_SUM = "analysis/outputs/year_qc/outside_year_summary.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Years to scan, e.g. --years 2021 2022 2023",
    )
    p.add_argument(
        "--in-dir",
        default=DEFAULT_IN_DIR,
        help=f"Input directory of outside station CSVs (default: {DEFAULT_IN_DIR})",
    )
    p.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for cleaned year files (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--out-sum",
        default=DEFAULT_OUT_SUM,
        help=f"Summary CSV path (default: {DEFAULT_OUT_SUM})",
    )
    p.add_argument(
        "--broken-cutoff",
        default="1980-01-01 00:00:00",
        help="Any parsed timestamp earlier than this is treated as broken time.",
    )
    return p.parse_args()


def ensure_dirs(out_dir: str, out_sum: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_sum), exist_ok=True)


def year_bounds(year: int):
    t0 = pd.Timestamp(f"{year}-01-01 00:00:00", tz=TZ)
    t1 = pd.Timestamp(f"{year}-12-31 23:00:00", tz=TZ)
    return t0, t1


def to_bkk(dt_series: pd.Series) -> pd.Series:
    """
    Parse timestamps and normalize to Asia/Bangkok.
    If parsed datetimes are naive, localize to Bangkok.
    If timezone-aware, convert to Bangkok.
    """
    dt = pd.to_datetime(dt_series, errors="coerce")

    if dt.notna().sum() == 0:
        return dt

    # pandas Series.dt.tz works for datetime-like series
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(TZ)
    else:
        dt = dt.dt.tz_convert(TZ)

    return dt


def keep_hourly_like(df: pd.DataFrame):
    info = {}

    df = df.copy()
    df["time"] = to_bkk(df["time"])

    info["time_ok"] = int(df["time"].notna().sum())
    df = df.dropna(subset=["time"]).copy()

    # Keep only rows exactly on the hour
    whole_hour = (df["time"].dt.minute == 0) & (df["time"].dt.second == 0)
    df = df.loc[whole_hour].copy()
    info["whole_hour_rows"] = int(len(df))

    keep_cols = [c for c in ["time", "pm25", "outside_station_id"] if c in df.columns]
    df = df[keep_cols].copy()

    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")
    info["unique_time_rows"] = int(len(df))

    if len(df) >= 2:
        diffs = df["time"].diff().dropna()
        step_1h_count = int((diffs == pd.Timedelta(hours=1)).sum())
        step_1d_count = int((diffs == pd.Timedelta(days=1)).sum())
        info["step_1h_count"] = step_1h_count
        info["step_1d_count"] = step_1d_count
        info["median_step"] = str(diffs.median())
        info["hourly_step_frac"] = float(step_1h_count / len(diffs))
        info["daily_step_frac"] = float(step_1d_count / len(diffs))
    else:
        info["step_1h_count"] = 0
        info["step_1d_count"] = 0
        info["median_step"] = ""
        info["hourly_step_frac"] = np.nan
        info["daily_step_frac"] = np.nan

    return df, info


def cut_window(df: pd.DataFrame, year: int) -> pd.DataFrame:
    t0, t1 = year_bounds(year)
    return df.loc[(df["time"] >= t0) & (df["time"] <= t1)].copy()


def true_run_lengths(mask: np.ndarray):
    """
    Returns lengths of contiguous True runs.
    """
    lengths = []
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
                cur = 0
    if cur > 0:
        lengths.append(cur)
    return lengths


def summarize_nonnull_runs(mask: np.ndarray):
    lengths = true_run_lengths(mask)

    if len(lengths) == 0:
        return {
            "nonnull_run_count": 0,
            "longest_nonnull_run_hours": 0,
            "longest_nonnull_run_days": 0.0,
            "median_nonnull_run_hours": 0.0,
        }

    longest = int(max(lengths))
    return {
        "nonnull_run_count": int(len(lengths)),
        "longest_nonnull_run_hours": longest,
        "longest_nonnull_run_days": float(longest / 24.0),
        "median_nonnull_run_hours": float(np.median(lengths)),
    }


def reindex_full_hours(df: pd.DataFrame, year: int):
    t0, t1 = year_bounds(year)
    full = pd.date_range(t0, t1, freq="h")

    tmp = df.set_index("time")[["pm25"]].copy()
    out = tmp.reindex(full)
    out.index.name = "time"
    out = out.reset_index()

    # Count negatives BEFORE replacing
    pm25_negative = int((out["pm25"] < 0).sum(skipna=True))

    # Replace negatives with NaN for wavelet-safe cleaning
    out.loc[out["pm25"] < 0, "pm25"] = np.nan

    mask = out["pm25"].notna().to_numpy()
    run_info = summarize_nonnull_runs(mask)

    info = {
        "expected_hours": int(len(out)),
        "pm25_na": int(out["pm25"].isna().sum()),
        "pm25_na_frac": float(out["pm25"].isna().mean()),
        "pm25_negative": pm25_negative,
        "pm25_nonnull": int(out["pm25"].notna().sum()),
        "coverage_frac_of_year": float(out["pm25"].notna().mean()),
        **run_info,
    }
    return out, info


def detect_broken_time(raw_time: pd.Series, broken_cutoff: pd.Timestamp):
    """
    Robust broken-time screen.
    If any parsed time is earlier than the cutoff, treat file as broken.
    For this project, anything pre-1980 is definitely invalid.
    """
    dt = to_bkk(raw_time)
    dt = dt.dropna()

    if len(dt) == 0:
        return False, np.nan, np.nan, 0

    tmin = dt.min()
    tmax = dt.max()
    n_old = int((dt < broken_cutoff).sum())

    is_broken = bool(n_old > 0)
    return is_broken, tmin, tmax, n_old


def prelim_tier(row: pd.Series) -> str:
    """
    Station-level preliminary tier only.
    Final wavelet tier must be decided later at PAIR level.
    """
    if row["status"] != "OK":
        return ""

    # If daily-heavy clearly dominates hourly steps, mark as Tier C immediately
    if pd.notna(row.get("step_1d_count", np.nan)) and pd.notna(row.get("step_1h_count", np.nan)):
        if row["step_1d_count"] > row["step_1h_count"]:
            return "Tier C"

    na_frac = row.get("pm25_na_frac", np.nan)
    longest = row.get("longest_nonnull_run_hours", 0)

    if pd.notna(na_frac) and na_frac <= 0.15 and longest >= 24 * 120:
        return "Tier A"

    if pd.notna(na_frac) and na_frac <= 0.40 and longest >= 24 * 60:
        return "Tier B"

    return "Tier C"


def append_same_status_for_all_years(rows, years, file_name, status, extra=None):
    extra = extra or {}
    for year in years:
        rows.append({
            "year": year,
            "file": file_name,
            "status": status,
            **extra
        })


def main():
    args = parse_args()
    years = sorted(set(args.years))
    in_dir = args.in_dir
    out_dir = args.out_dir
    out_sum = args.out_sum
    broken_cutoff = pd.Timestamp(args.broken_cutoff, tz=TZ)

    ensure_dirs(out_dir, out_sum)

    rows = []
    paths = sorted(glob.glob(os.path.join(in_dir, "outside_*.csv")))

    if len(paths) == 0:
        print(f"[WARN] No files found in: {in_dir}")
        return

    for p in paths:
        name = os.path.basename(p)

        try:
            raw = pd.read_csv(p)
        except Exception as e:
            append_same_status_for_all_years(
                rows, years, name, "READ_FAIL", {"error": str(e)}
            )
            continue

        if not {"time", "pm25"}.issubset(set(raw.columns)):
            append_same_status_for_all_years(
                rows, years, name, "MISSING_COLS", {"cols": "|".join(raw.columns)}
            )
            continue

        is_broken, raw_tmin, raw_tmax, n_old = detect_broken_time(raw["time"], broken_cutoff)
        if is_broken:
            append_same_status_for_all_years(
                rows, years, name, "BROKEN_TIME_1970",
                {
                    "raw_time_min": str(raw_tmin),
                    "raw_time_max": str(raw_tmax),
                    "n_times_before_cutoff": n_old,
                    "note": "re-ingest from Excel needed"
                }
            )
            continue

        cleaned, info1 = keep_hourly_like(raw)

        for year in years:
            windowed = cut_window(cleaned, year)

            if len(windowed) == 0:
                rows.append({
                    "year": year,
                    "file": name,
                    "status": "NO_DATA_IN_WINDOW",
                    "raw_time_min": str(raw_tmin),
                    "raw_time_max": str(raw_tmax),
                    **info1
                })
                continue

            grid, info2 = reindex_full_hours(windowed, year)

            year_dir = os.path.join(out_dir, str(year))
            os.makedirs(year_dir, exist_ok=True)

            out_name = name.replace(".csv", f"_{year}_hourly.csv")
            out_path = os.path.join(year_dir, out_name)
            grid.to_csv(out_path, index=False)

            row = {
                "year": year,
                "file": name,
                "status": "OK",
                "out_file": out_path,
                "raw_time_min": str(raw_tmin),
                "raw_time_max": str(raw_tmax),
                **info1,
                **info2
            }
            row["tier_prelim"] = prelim_tier(pd.Series(row))
            rows.append(row)

    dfsum = pd.DataFrame(rows)

    # Nice column order if present
    preferred_cols = [
        "year", "file", "status", "tier_prelim", "out_file",
        "raw_time_min", "raw_time_max", "n_times_before_cutoff",
        "time_ok", "whole_hour_rows", "unique_time_rows",
        "step_1h_count", "step_1d_count", "hourly_step_frac", "daily_step_frac", "median_step",
        "expected_hours", "pm25_nonnull", "pm25_na", "pm25_na_frac", "coverage_frac_of_year",
        "pm25_negative",
        "nonnull_run_count", "longest_nonnull_run_hours", "longest_nonnull_run_days", "median_nonnull_run_hours",
        "note", "error", "cols"
    ]
    keep_cols = [c for c in preferred_cols if c in dfsum.columns] + [c for c in dfsum.columns if c not in preferred_cols]
    dfsum = dfsum[keep_cols]

    dfsum.to_csv(out_sum, index=False)
    print(f"[DONE] wrote {out_sum}")

    # Console summaries
    print("\n=== STATUS COUNTS BY YEAR ===")
    try:
        status_tab = (
            dfsum.groupby(["year", "status"])
                 .size()
                 .reset_index(name="count")
                 .sort_values(["year", "status"])
        )
        print(status_tab.to_string(index=False))
    except Exception:
        pass

    print("\n=== PRELIMINARY TIER COUNTS (OK rows only) ===")
    ok = dfsum[dfsum["status"] == "OK"].copy()
    if len(ok) > 0 and "tier_prelim" in ok.columns:
        tier_tab = (
            ok.groupby(["year", "tier_prelim"])
              .size()
              .reset_index(name="count")
              .sort_values(["year", "tier_prelim"])
        )
        print(tier_tab.to_string(index=False))

        print("\n=== BEST CANDIDATES BY YEAR (lowest NA first) ===")
        for year in sorted(ok["year"].dropna().unique()):
            tmp = ok[ok["year"] == year].copy().sort_values(
                ["pm25_na_frac", "longest_nonnull_run_hours"],
                ascending=[True, False]
            )
            cols = [
                "file", "tier_prelim", "pm25_na_frac", "pm25_nonnull",
                "longest_nonnull_run_hours", "step_1h_count", "step_1d_count", "out_file"
            ]
            cols = [c for c in cols if c in tmp.columns]
            print(f"\n--- YEAR {int(year)} ---")
            print(tmp[cols].head(12).to_string(index=False))
    else:
        print("[WARN] No OK station-year files found.")


if __name__ == "__main__":
    main()