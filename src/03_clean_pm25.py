#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean park hourly PM2.5 time series.

Input:
  analysis/data_processed/parks_hourly/parkXX.csv

Output:
  analysis/data_processed/parks_hourly_clean/parkXX.csv

Rules (safe default):
  - pm25 < 0  -> NaN
  - keep everything else unchanged
Also writes:
  analysis/outputs/qc_parks_hourly_clean_summary.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "analysis" / "data_processed" / "parks_hourly"
OUT_DIR = ROOT / "analysis" / "data_processed" / "parks_hourly_clean"
QC_OUT = ROOT / "analysis" / "outputs" / "qc_parks_hourly_clean_summary.csv"

TZ = "Asia/Bangkok"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(IN_DIR.glob("park*.csv")):
        df = pd.read_csv(f)

        # parse datetime (your saved 't' is timezone-aware text)
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(TZ)

        # numeric pm25
        df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

        n0 = len(df)
        n_neg = int((df["pm25"] < 0).sum())

        # clean
        df.loc[df["pm25"] < 0, "pm25"] = pd.NA

        # keep rows (do not drop), so timestamps remain consistent for later gap logic
        out = OUT_DIR / f.name
        df.sort_values("t").to_csv(out, index=False, encoding="utf-8-sig")

        rows.append({
            "park_file": f.name,
            "n_rows": n0,
            "n_negative_set_nan": n_neg,
            "pm25_min_after": df["pm25"].min(skipna=True),
            "pm25_max_after": df["pm25"].max(skipna=True),
            "n_missing_pm25_after": int(df["pm25"].isna().sum()),
            "t_min": df["t"].min(),
            "t_max": df["t"].max(),
        })

        print(f"[OK] {f.name}: set {n_neg} negatives to NaN -> {out}")

    pd.DataFrame(rows).to_csv(QC_OUT, index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {QC_OUT}")

if __name__ == "__main__":
    main()
