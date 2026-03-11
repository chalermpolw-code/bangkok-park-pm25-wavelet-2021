#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import pandas as pd
import numpy as np

TZ = "Asia/Bangkok"

# ===== choose your wavelet window here =====
WINDOW_START = "2021-01-01 00:00:00"
WINDOW_END   = "2021-12-31 23:00:00"
# ==========================================

IN_DIR  = "analysis/data_processed/outside_hourly"
OUT_DIR = "analysis/data_processed/outside_hourly_clean"
OUT_SUM = "analysis/outputs/outside_hourly_clean_summary.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUT_SUM), exist_ok=True)

def to_bkk(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce")
    # localize/convert timezone robustly
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(TZ)
    else:
        dt = dt.dt.tz_convert(TZ)
    return dt

def keep_hourly_like(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {}

    dt = to_bkk(df["time"])
    df = df.copy()
    df["time"] = dt

    info["time_ok"] = int(df["time"].notna().sum())
    df = df.dropna(subset=["time"])

    # keep only whole-hour stamps
    whole_hour = (df["time"].dt.minute == 0) & (df["time"].dt.second == 0)
    df = df.loc[whole_hour].copy()
    info["whole_hour_rows"] = int(len(df))

    keep_cols = [c for c in ["time", "pm25", "outside_station_id"] if c in df.columns]
    df = df[keep_cols].copy()

    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")
    info["unique_time_rows"] = int(len(df))

    if len(df) >= 2:
        diffs = df["time"].diff().dropna()
        info["step_1h_count"] = int((diffs == pd.Timedelta(hours=1)).sum())
        info["step_1d_count"] = int((diffs == pd.Timedelta(days=1)).sum())
        info["median_step"] = str(diffs.median())
    else:
        info["step_1h_count"] = 0
        info["step_1d_count"] = 0
        info["median_step"] = ""

    return df, info

def cut_window(df: pd.DataFrame) -> pd.DataFrame:
    t0 = pd.Timestamp(WINDOW_START, tz=TZ)
    t1 = pd.Timestamp(WINDOW_END, tz=TZ)
    return df.loc[(df["time"] >= t0) & (df["time"] <= t1)].copy()

def reindex_full_hours(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    t0 = pd.Timestamp(WINDOW_START, tz=TZ)
    t1 = pd.Timestamp(WINDOW_END, tz=TZ)
    full = pd.date_range(t0, t1, freq="h")

    tmp = df.set_index("time")[["pm25"]].copy()
    out = tmp.reindex(full)
    out.index.name = "time"
    out = out.reset_index()

    info = {
        "n_hours": int(len(out)),
        "pm25_na": int(out["pm25"].isna().sum()),
        "pm25_na_frac": float(out["pm25"].isna().mean()),
        "pm25_negative": int((out["pm25"] < 0).sum(skipna=True)),
        "pm25_nonnull": int(out["pm25"].notna().sum()),
    }
    return out, info


rows = []
paths = sorted(glob.glob(os.path.join(IN_DIR, "outside_*.csv")))

# tz-aware cutoff for the "1970 broken" detection
BROKEN_CUTOFF = pd.Timestamp("1980-01-01", tz=TZ)

for p in paths:
    name = os.path.basename(p)

    try:
        raw = pd.read_csv(p)
    except Exception as e:
        rows.append({"file": name, "status": "READ_FAIL", "error": str(e)})
        continue

    if not {"time", "pm25"}.issubset(set(raw.columns)):
        rows.append({"file": name, "status": "MISSING_COLS", "cols": "|".join(raw.columns)})
        continue

    # Detect the 1970-ns broken case robustly (tz-safe)
    dt_preview = pd.to_datetime(raw["time"].head(200), errors="coerce")
    dt_preview = dt_preview.dropna()
    if len(dt_preview) > 0:
        # convert preview to Bangkok tz (or localize if naive)
        if dt_preview.dt.tz is None:
            dt_preview = dt_preview.dt.tz_localize(TZ)
        else:
            dt_preview = dt_preview.dt.tz_convert(TZ)

        if dt_preview.min() < BROKEN_CUTOFF:
            rows.append({"file": name, "status": "BROKEN_TIME_1970", "note": "re-ingest from Excel needed"})
            continue

    cleaned, info1 = keep_hourly_like(raw)
    windowed = cut_window(cleaned)

    if len(windowed) == 0:
        rows.append({"file": name, "status": "NO_DATA_IN_WINDOW", **info1})
        continue

    grid, info2 = reindex_full_hours(windowed)

    out_name = name.replace(".csv", "_2021_hourly.csv")
    out_path = os.path.join(OUT_DIR, out_name)
    grid.to_csv(out_path, index=False)

    rows.append({
        "file": name,
        "status": "OK",
        "out_file": out_path,
        "time_ok": info1.get("time_ok", 0),
        "whole_hour_rows": info1.get("whole_hour_rows", 0),
        "unique_time_rows": info1.get("unique_time_rows", 0),
        "step_1h_count": info1.get("step_1h_count", 0),
        "step_1d_count": info1.get("step_1d_count", 0),
        "median_step": info1.get("median_step", ""),
        **info2
    })

dfsum = pd.DataFrame(rows)
dfsum.to_csv(OUT_SUM, index=False)
print(f"[DONE] wrote {OUT_SUM}")

# quick console view: best ones first
ok = dfsum[dfsum["status"] == "OK"].copy()
if len(ok) > 0:
    ok = ok.sort_values("pm25_na_frac")
    print(ok[["file","pm25_na_frac","pm25_nonnull","pm25_negative","step_1h_count","step_1d_count","out_file"]].to_string(index=False))
else:
    print("[WARN] No OK files in this window.")
