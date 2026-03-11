#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

PAIRS = "analysis/outputs/pairs_2021_tierA.csv"

INSIDE_DIR = "analysis/data_processed/parks_hourly_grid"          # parkXX.csv (full 2021–2023 grid)
OUTSIDE_DIR = "analysis/data_processed/outside_hourly_clean"      # outside_YY_2021_hourly.csv (2021 grid)

OUT_DIR = "analysis/data_processed/pairs_2021"
OUT_SUM = "analysis/outputs/pairs_2021_summary.csv"

TZ = "Asia/Bangkok"
START = pd.Timestamp("2021-01-01 00:00:00", tz=TZ)
END   = pd.Timestamp("2021-12-31 23:00:00", tz=TZ)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_SUM), exist_ok=True)

def read_inside(park_id: int) -> pd.DataFrame:
    fp = f"{INSIDE_DIR}/park{park_id:02d}.csv"
    df = pd.read_csv(fp)
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    # ensure tz
    if df["t"].dt.tz is None:
        df["t"] = df["t"].dt.tz_localize(TZ)
    else:
        df["t"] = df["t"].dt.tz_convert(TZ)
    df["pm25_in"] = pd.to_numeric(df["pm25"], errors="coerce")
    return df[["t","pm25_in"]]

def read_outside(outside_id: int) -> pd.DataFrame:
    fp = f"{OUTSIDE_DIR}/outside_{outside_id:02d}_2021_hourly.csv"
    df = pd.read_csv(fp)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(TZ)
    else:
        df["time"] = df["time"].dt.tz_convert(TZ)
    df["pm25_out"] = pd.to_numeric(df["pm25"], errors="coerce")
    return df.rename(columns={"time":"t"})[["t","pm25_out"]]

def main():
    pairs = pd.read_csv(PAIRS)

    summary_rows = []

    for _, r in pairs.iterrows():
        park_id = int(r["park_id"])
        outside_id = int(r["outside_station_id"])

        inside = read_inside(park_id)
        outside = read_outside(outside_id)

        # clip to 2021
        inside = inside[(inside["t"] >= START) & (inside["t"] <= END)].copy()
        outside = outside[(outside["t"] >= START) & (outside["t"] <= END)].copy()

        # merge on t
        m = pd.merge(inside, outside, on="t", how="inner")

        # overlap coverage metrics
        n_hours = len(m)
        n_in = int(m["pm25_in"].notna().sum())
        n_out = int(m["pm25_out"].notna().sum())
        n_both = int((m["pm25_in"].notna() & m["pm25_out"].notna()).sum())

        # expected hours in 2021 (non-leap year)
        expected = 8760
        frac_both = n_both / expected

        out_fp = f"{OUT_DIR}/park{park_id:02d}_outside{outside_id:02d}_2021.csv"
        m.to_csv(out_fp, index=False, encoding="utf-8-sig")

        summary_rows.append({
            "park_id": park_id,
            "outside_station_id": outside_id,
            "distance_km": float(r["distance_km"]),
            "rows_merged": n_hours,
            "n_in_nonnull": n_in,
            "n_out_nonnull": n_out,
            "n_both_nonnull": n_both,
            "both_coverage_frac_of_8760": frac_both,
            "out_file": out_fp
        })

        print(f"[OK] park{park_id:02d} + outside{outside_id:02d}: both={n_both}/{expected} ({frac_both:.3f}) -> {out_fp}")

    summ = pd.DataFrame(summary_rows).sort_values("both_coverage_frac_of_8760", ascending=False)
    summ.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {OUT_SUM}")

if __name__ == "__main__":
    main()