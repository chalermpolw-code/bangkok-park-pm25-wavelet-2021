#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pair-level overlap QC for park–outside replication years.

Purpose
-------
Confirm whether any year-specific park–outside pairs are usable for wavelet analysis.

This script is especially useful after station-level scan results suggest that
2022/2023 are sparse or fragmented.

Default behavior
----------------
- Uses deterministic nearest-station mapping (rank <= 1 by default)
- Reads park hourly grid files
- Reads year-specific outside cleaned files from yearscan
- Clips park data to each year
- Merges park + outside on hourly timestamps
- Computes pair overlap metrics
- Assigns a preliminary pair tier:
    Tier A = main wavelet candidate
    Tier B = sensitivity only
    Tier C = not suitable
"""

import os
import argparse
import numpy as np
import pandas as pd

TZ = "Asia/Bangkok"

DEFAULT_MAPPING = "analysis/data_processed/park_to_outside_station_top3.csv"
DEFAULT_PARK_DIR = "analysis/data_processed/parks_hourly_grid"
DEFAULT_OUTSIDE_DIR = "analysis/data_processed/outside_hourly_clean_yearscan"
DEFAULT_OUTSIDE_SUM = "analysis/outputs/year_qc/outside_year_summary.csv"
DEFAULT_OUT_SUM = "analysis/outputs/year_qc/pair_year_overlap_summary.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2021, 2022, 2023],
        help="Years to check, e.g. --years 2021 2022 2023",
    )
    p.add_argument(
        "--mapping",
        default=DEFAULT_MAPPING,
        help=f"Park→outside mapping CSV (default: {DEFAULT_MAPPING})",
    )
    p.add_argument(
        "--park-dir",
        default=DEFAULT_PARK_DIR,
        help=f"Directory of park hourly grid CSVs (default: {DEFAULT_PARK_DIR})",
    )
    p.add_argument(
        "--outside-dir",
        default=DEFAULT_OUTSIDE_DIR,
        help=f"Directory containing year folders of cleaned outside CSVs (default: {DEFAULT_OUTSIDE_DIR})",
    )
    p.add_argument(
        "--outside-summary",
        default=DEFAULT_OUTSIDE_SUM,
        help=f"Outside year summary CSV (default: {DEFAULT_OUTSIDE_SUM})",
    )
    p.add_argument(
        "--out-sum",
        default=DEFAULT_OUT_SUM,
        help=f"Output pair summary CSV (default: {DEFAULT_OUT_SUM})",
    )
    p.add_argument(
        "--max-rank",
        type=int,
        default=1,
        help="Maximum mapping rank to include (default: 1 for strict replication)",
    )
    return p.parse_args()


def ensure_dirs(out_sum: str):
    os.makedirs(os.path.dirname(out_sum), exist_ok=True)


def year_bounds(year: int):
    t0 = pd.Timestamp(f"{year}-01-01 00:00:00", tz=TZ)
    t1 = pd.Timestamp(f"{year}-12-31 23:00:00", tz=TZ)
    return t0, t1


def to_bkk(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce")
    if dt.notna().sum() == 0:
        return dt
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(TZ)
    else:
        dt = dt.dt.tz_convert(TZ)
    return dt


def normalize_colname(c: str) -> str:
    return (
        str(c).strip().lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
    )


def find_time_col(cols):
    candidates = {
        "t", "time", "datetime", "timestamp", "datehour", "dateandtime"
    }
    for c in cols:
        if normalize_colname(c) in candidates:
            return c
    return None


def find_pm25_col(cols):
    candidates = {
        "pm25",
        "pm25ugm3",
        "pm25ugm^3",
        "pm25ugm",
        "pm25conc",
        "pm25value",
        "pm25mean",
        "pm25average",
        "pm25avg",
        "pm25hr",
        "pm25hourly",
        "pm25mass",
        "pm25µgm3",
        "pm25μgm3",
        "pm25ugm-3",
    }
    for c in cols:
        if normalize_colname(c) in candidates:
            return c
    return None


def cut_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    t0, t1 = year_bounds(year)
    return df.loc[(df["time"] >= t0) & (df["time"] <= t1)].copy()


def true_run_lengths(mask: np.ndarray):
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


def summarize_runs(mask: np.ndarray, prefix: str):
    lengths = true_run_lengths(mask)
    if len(lengths) == 0:
        return {
            f"{prefix}_run_count": 0,
            f"longest_{prefix}_run_hours": 0,
            f"longest_{prefix}_run_days": 0.0,
            f"median_{prefix}_run_hours": 0.0,
        }

    longest = int(max(lengths))
    return {
        f"{prefix}_run_count": int(len(lengths)),
        f"longest_{prefix}_run_hours": longest,
        f"longest_{prefix}_run_days": float(longest / 24.0),
        f"median_{prefix}_run_hours": float(np.median(lengths)),
    }


def pair_tier_prelim(row: pd.Series) -> str:
    if row["status"] != "OK":
        return ""

    both_cov = row.get("both_coverage_frac_of_year", np.nan)
    longest_both = row.get("longest_both_nonnull_run_hours", 0)

    if pd.notna(both_cov) and both_cov >= 0.85 and longest_both >= 24 * 120:
        return "Tier A"

    if pd.notna(both_cov) and both_cov >= 0.60 and longest_both >= 24 * 60:
        return "Tier B"

    return "Tier C"


def load_outside_summary(path: str):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def main():
    args = parse_args()
    years = sorted(set(args.years))

    ensure_dirs(args.out_sum)

    mapping = pd.read_csv(args.mapping)

    needed = {"park_id", "outside_station_id", "rank"}
    if not needed.issubset(mapping.columns):
        raise ValueError(f"Mapping file must contain columns: {sorted(needed)}")

    mapping = mapping.loc[mapping["rank"] <= args.max_rank].copy()
    mapping["park_id"] = mapping["park_id"].astype(int)
    mapping["outside_station_id"] = mapping["outside_station_id"].astype(int)
    mapping["rank"] = mapping["rank"].astype(int)

    outside_sum = load_outside_summary(args.outside_summary)
    if len(outside_sum):
        outside_sum["year"] = outside_sum["year"].astype(int)
        tmp = outside_sum["file"].astype(str).str.extract(r"outside_(\d+)\.csv")
        outside_sum["outside_station_id"] = pd.to_numeric(tmp[0], errors="coerce").astype("Int64")

    rows = []

    for _, m in mapping.iterrows():
        park_id = int(m["park_id"])
        outside_id = int(m["outside_station_id"])
        rank = int(m["rank"])

        pair_stub = f"park{park_id:02d}_outside{outside_id:02d}"
        park_path = os.path.join(args.park_dir, f"park{park_id:02d}.csv")

        if not os.path.exists(park_path):
            for year in years:
                rows.append({
                    "year": year,
                    "pair_id": f"{pair_stub}_{year}",
                    "park_id": park_id,
                    "outside_station_id": outside_id,
                    "rank": rank,
                    "status": "PARK_FILE_MISSING",
                    "park_file": park_path,
                })
            continue

        park = pd.read_csv(park_path)

        time_col = find_time_col(park.columns)
        pm25_col = find_pm25_col(park.columns)

        if time_col is None or pm25_col is None:
            for year in years:
                rows.append({
                    "year": year,
                    "pair_id": f"{pair_stub}_{year}",
                    "park_id": park_id,
                    "outside_station_id": outside_id,
                    "rank": rank,
                    "status": "PARK_MISSING_COLS",
                    "park_file": park_path,
                    "park_cols": "|".join(map(str, park.columns)),
                    "park_time_col_detected": time_col if time_col is not None else "",
                    "park_pm25_col_detected": pm25_col if pm25_col is not None else "",
                })
            continue

        park = park.rename(columns={time_col: "time", pm25_col: "pm25"}).copy()
        park["time"] = to_bkk(park["time"])
        park["pm25"] = pd.to_numeric(park["pm25"], errors="coerce")
        park.loc[park["pm25"] < 0, "pm25"] = np.nan
        park = park.dropna(subset=["time"]).sort_values("time")
        park = park.drop_duplicates(subset=["time"], keep="last")

        for year in years:
            out_path = os.path.join(
                args.outside_dir, str(year), f"outside_{outside_id:02d}_{year}_hourly.csv"
            )

            base = {
                "year": year,
                "pair_id": f"{pair_stub}_{year}",
                "park_id": park_id,
                "outside_station_id": outside_id,
                "rank": rank,
                "park_file": park_path,
                "outside_file": out_path,
            }

            if not os.path.exists(out_path):
                rows.append({**base, "status": "OUTSIDE_FILE_MISSING"})
                continue

            outside = pd.read_csv(out_path)

            if not {"time", "pm25"}.issubset(outside.columns):
                rows.append({
                    **base,
                    "status": "OUTSIDE_MISSING_COLS",
                    "outside_cols": "|".join(map(str, outside.columns)),
                })
                continue

            outside = outside.copy()
            outside["time"] = to_bkk(outside["time"])
            outside["pm25"] = pd.to_numeric(outside["pm25"], errors="coerce")
            outside.loc[outside["pm25"] < 0, "pm25"] = np.nan
            outside = outside.dropna(subset=["time"]).sort_values("time")
            outside = outside.drop_duplicates(subset=["time"], keep="last")

            park_y = cut_year(park, year)[["time", "pm25"]].rename(columns={"pm25": "pm25_park"})
            out_y = cut_year(outside, year)[["time", "pm25"]].rename(columns={"pm25": "pm25_outside"})

            if len(park_y) == 0:
                rows.append({**base, "status": "NO_PARK_DATA_IN_YEAR"})
                continue

            if len(out_y) == 0:
                rows.append({**base, "status": "NO_OUTSIDE_DATA_IN_YEAR"})
                continue

            merged = pd.merge(park_y, out_y, on="time", how="outer").sort_values("time")

            t0, t1 = year_bounds(year)
            full = pd.DataFrame({"time": pd.date_range(t0, t1, freq="h")})
            merged = full.merge(merged, on="time", how="left")

            park_mask = merged["pm25_park"].notna().to_numpy()
            out_mask = merged["pm25_outside"].notna().to_numpy()
            both_mask = (merged["pm25_park"].notna() & merged["pm25_outside"].notna()).to_numpy()

            row = {
                **base,
                "status": "OK",
                "expected_hours": int(len(merged)),
                "park_nonnull_hours": int(park_mask.sum()),
                "outside_nonnull_hours": int(out_mask.sum()),
                "both_nonnull_hours": int(both_mask.sum()),
                "park_coverage_frac_of_year": float(park_mask.mean()),
                "outside_coverage_frac_of_year": float(out_mask.mean()),
                "both_coverage_frac_of_year": float(both_mask.mean()),
                **summarize_runs(park_mask, "park_nonnull"),
                **summarize_runs(out_mask, "outside_nonnull"),
                **summarize_runs(both_mask, "both_nonnull"),
            }

            if len(outside_sum):
                tmp = outside_sum[
                    (outside_sum["year"] == year) &
                    (outside_sum["outside_station_id"] == outside_id)
                ]
                if len(tmp):
                    tmp = tmp.iloc[0]
                    row["outside_status_yearscan"] = tmp.get("status", "")
                    row["outside_tier_prelim_yearscan"] = tmp.get("tier_prelim", "")
                    row["outside_pm25_na_frac_yearscan"] = tmp.get("pm25_na_frac", np.nan)
                    row["outside_longest_nonnull_run_hours_yearscan"] = tmp.get(
                        "longest_nonnull_run_hours", np.nan
                    )

            row["pair_tier_prelim"] = pair_tier_prelim(pd.Series(row))
            rows.append(row)

    df = pd.DataFrame(rows)

    preferred = [
        "year", "pair_id", "status", "pair_tier_prelim",
        "park_id", "outside_station_id", "rank",
        "expected_hours",
        "park_nonnull_hours", "outside_nonnull_hours", "both_nonnull_hours",
        "park_coverage_frac_of_year", "outside_coverage_frac_of_year", "both_coverage_frac_of_year",
        "longest_park_nonnull_run_hours", "longest_outside_nonnull_run_hours", "longest_both_nonnull_run_hours",
        "longest_park_nonnull_run_days", "longest_outside_nonnull_run_days", "longest_both_nonnull_run_days",
        "median_park_nonnull_run_hours", "median_outside_nonnull_run_hours", "median_both_nonnull_run_hours",
        "outside_status_yearscan", "outside_tier_prelim_yearscan",
        "outside_pm25_na_frac_yearscan", "outside_longest_nonnull_run_hours_yearscan",
        "park_file", "outside_file",
        "park_cols", "outside_cols",
        "park_time_col_detected", "park_pm25_col_detected",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(args.out_sum, index=False)
    print(f"[DONE] wrote {args.out_sum}")

    print("\n=== STATUS COUNTS BY YEAR ===")
    tab1 = (
        df.groupby(["year", "status"])
          .size()
          .reset_index(name="count")
          .sort_values(["year", "status"])
    )
    print(tab1.to_string(index=False))

    ok = df[df["status"] == "OK"].copy()
    if len(ok):
        print("\n=== PAIR TIER COUNTS (OK rows only) ===")
        tab2 = (
            ok.groupby(["year", "pair_tier_prelim"])
              .size()
              .reset_index(name="count")
              .sort_values(["year", "pair_tier_prelim"])
        )
        print(tab2.to_string(index=False))

        print("\n=== BEST PAIRS BY YEAR ===")
        for year in sorted(ok["year"].unique()):
            tmp = ok[ok["year"] == year].copy().sort_values(
                ["both_coverage_frac_of_year", "longest_both_nonnull_run_hours"],
                ascending=[False, False]
            )
            show_cols = [
                "pair_id", "pair_tier_prelim",
                "both_coverage_frac_of_year", "both_nonnull_hours",
                "longest_both_nonnull_run_hours",
                "park_nonnull_hours", "outside_nonnull_hours",
                "outside_tier_prelim_yearscan"
            ]
            show_cols = [c for c in show_cols if c in tmp.columns]
            print(f"\n--- YEAR {int(year)} ---")
            print(tmp[show_cols].head(15).to_string(index=False))
    else:
        print("[WARN] No OK merged pairs found.")


if __name__ == "__main__":
    main()