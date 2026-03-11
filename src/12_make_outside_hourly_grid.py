#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_make_outside_hourly_grid.py

Build a fixed hourly grid (Asia/Bangkok) for each outside station CSV:
  input : analysis/data_processed/outside_hourly/outside_XX.csv
          columns required: time, pm25, outside_station_id
  output: analysis/data_processed/outside_hourly_grid/outside_XX_grid.csv
          columns: time, pm25, outside_station_id
  summary: analysis/outputs/outside_hourly_grid_summary.csv

Notes:
- Keeps your original values, just reindexes onto full hourly timeline.
- Handles timezone safely (localize/convert to Asia/Bangkok).
- If file is daily (00:00 only), it will still run but with huge NA fraction.
"""

from __future__ import annotations

import os
import glob
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
TZ = "Asia/Bangkok"
START = "2021-01-01 00:00:00"
END = "2023-12-31 23:00:00"
FREQ = "h"

IN_DIR = "analysis/data_processed/outside_hourly"
OUT_DIR = "analysis/data_processed/outside_hourly_grid"
SUMMARY_PATH = "analysis/outputs/outside_hourly_grid_summary.csv"

DT_COL = "time"
PM_COL = "pm25"
ID_COL = "outside_station_id"

# Full hourly index (timezone-aware)
FULL_INDEX = pd.date_range(START, END, freq=FREQ, tz=TZ)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)


def _coerce_time_to_bkk(series: pd.Series) -> pd.Series:
    """Parse datetime and ensure tz=Asia/Bangkok."""
    dt = pd.to_datetime(series, errors="coerce")

    # If parsed datetimes are tz-naive -> localize to Bangkok
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(TZ)
    else:
        # tz-aware -> convert to Bangkok
        dt = dt.dt.tz_convert(TZ)

    return dt


def build_grid_one(path: str) -> dict:
    """Build hourly grid for a single outside CSV. Returns summary dict."""
    base = os.path.basename(path)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {
            "file": base,
            "status": f"ERROR_READ:{type(e).__name__}",
            "n_hours": 0,
            "pm25_na": 0,
            "pm25_na_frac": 1.0,
        }

    # Validate required columns
    missing = [c for c in (DT_COL, PM_COL, ID_COL) if c not in df.columns]
    if missing:
        return {
            "file": base,
            "status": f"SKIP_MISSING_COLS:{','.join(missing)}",
            "n_hours": 0,
            "pm25_na": 0,
            "pm25_na_frac": 1.0,
        }

    # Parse & timezone-align
    df = df[[DT_COL, PM_COL, ID_COL]].copy()
    df[DT_COL] = _coerce_time_to_bkk(df[DT_COL])
    df = df.dropna(subset=[DT_COL])

    # Coerce pm25 numeric (keep NaNs)
    df[PM_COL] = pd.to_numeric(df[PM_COL], errors="coerce")

    # If duplicates exist, keep the last (safe choice)
    df = df.sort_values(DT_COL)
    df = df.drop_duplicates(subset=[DT_COL], keep="last")

    # Reindex to full hourly grid
    s = df.set_index(DT_COL)[PM_COL]
    grid_pm = s.reindex(FULL_INDEX)

    # Determine station id (should be constant)
    try:
        station_id = int(pd.to_numeric(df[ID_COL], errors="coerce").dropna().iloc[0])
    except Exception:
        station_id = None

    out = pd.DataFrame(
        {
            DT_COL: FULL_INDEX,
            PM_COL: grid_pm.values,
            ID_COL: station_id,
        }
    )

    out_path = os.path.join(
        OUT_DIR, base.replace(".csv", "_grid.csv")
    )
    out.to_csv(out_path, index=False)

    pm25_na = int(out[PM_COL].isna().sum())
    n_hours = int(len(out))
    pm25_na_frac = float(pm25_na / n_hours) if n_hours else 1.0

    return {
        "file": base,
        "status": "OK",
        "n_hours": n_hours,
        "pm25_na": pm25_na,
        "pm25_na_frac": pm25_na_frac,
        "out_file": os.path.basename(out_path),
    }


def main() -> None:
    files = sorted(glob.glob(os.path.join(IN_DIR, "outside_*.csv")))

    if not files:
        raise SystemExit(f"No files found in {IN_DIR}. Expected outside_*.csv")

    rows = []
    for path in files:
        rows.append(build_grid_one(path))

    summary = pd.DataFrame(rows)

    # If some rows don't have out_file, fill for stable CSV schema
    if "out_file" not in summary.columns:
        summary["out_file"] = ""

    # Nice ordering
    cols = ["file", "status", "n_hours", "pm25_na", "pm25_na_frac", "out_file"]
    summary = summary.reindex(columns=[c for c in cols if c in summary.columns])

    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"[DONE] wrote {SUMMARY_PATH}")
    print(summary.sort_values(["status", "pm25_na_frac", "file"]).head(30).to_string(index=False))


if __name__ == "__main__":
    main()

