#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "analysis" / "data_processed" / "parks_hourly"
OUT_DIR = ROOT / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    rows = []
    for f in sorted(IN_DIR.glob("park*.csv")):
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("Asia/Bangkok")
        df = df.sort_values("t")

        tmin = df["t"].min()
        tmax = df["t"].max()

        # expected hourly grid
        full = pd.date_range(tmin.floor("H"), tmax.ceil("H"), freq="H", tz="Asia/Bangkok")
        have = pd.DatetimeIndex(df["t"])
        missing = full.difference(have)

        rows.append({
            "park_file": f.name,
            "n_rows": len(df),
            "t_min": tmin,
            "t_max": tmax,
            "expected_hours": len(full),
            "missing_hours": len(missing),
            "missing_frac": (len(missing) / len(full)) if len(full) else None,
            "pm25_min": pd.to_numeric(df["pm25"], errors="coerce").min(),
            "pm25_max": pd.to_numeric(df["pm25"], errors="coerce").max(),
        })

    out = pd.DataFrame(rows).sort_values("park_file")
    out.to_csv(OUT_DIR / "qc_parks_hourly_summary.csv", index=False, encoding="utf-8-sig")
    print("[DONE] wrote analysis/outputs/qc_parks_hourly_summary.csv")

if __name__ == "__main__":
    main()

