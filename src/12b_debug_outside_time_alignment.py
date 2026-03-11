#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import glob
import os

TZ = "Asia/Bangkok"
START = "2021-01-01 00:00:00"
END   = "2023-12-31 23:00:00"
FULL_INDEX = pd.date_range(START, END, freq="h", tz=TZ)

IN_DIR = "analysis/data_processed/outside_hourly"

def to_bkk(x: pd.Series) -> pd.Series:
    dt = pd.to_datetime(x, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(TZ)
    else:
        dt = dt.dt.tz_convert(TZ)
    return dt

paths = sorted(glob.glob(os.path.join(IN_DIR, "outside_*.csv")))
print(f"Found {len(paths)} files")

for p in paths:
    name = os.path.basename(p)
    df = pd.read_csv(p)

    if "time" not in df.columns or "pm25" not in df.columns:
        print(name, "=> missing cols", df.columns.tolist())
        continue

    dt = to_bkk(df["time"])
    ok = dt.notna().sum()

    if ok == 0:
        print(name, "=> ALL time parsed as NaT (0 valid timestamps)")
        continue

    dt_ok = dt.dropna()
    in_range = ((dt_ok >= FULL_INDEX[0]) & (dt_ok <= FULL_INDEX[-1])).sum()

    # how many timestamps exactly hit the FULL_INDEX
    hit = dt_ok.isin(FULL_INDEX).sum()

    # detect frequency (hourly vs daily-ish)
    dt_unique = dt_ok.drop_duplicates().sort_values()
    diffs = dt_unique.diff().dropna()
    most_common_step = diffs.value_counts().head(3)

    print(
        f"{name} | n={len(df)} time_ok={ok} in_range={in_range} "
        f"hit_fullindex={hit} min={dt_ok.min()} max={dt_ok.max()} "
        f"top_steps={most_common_step.to_dict()}"
    )
