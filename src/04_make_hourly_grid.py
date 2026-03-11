#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "analysis" / "data_processed" / "parks_hourly_clean"
OUT_DIR = ROOT / "analysis" / "data_processed" / "parks_hourly_grid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TZ = "Asia/Bangkok"
START = pd.Timestamp("2021-01-01 00:00", tz=TZ)
END   = pd.Timestamp("2023-12-31 23:00", tz=TZ)

def main():
    full = pd.date_range(START, END, freq="h", tz=TZ)

    for f in sorted(IN_DIR.glob("park*.csv")):
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(TZ)
        df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
        pid = df["park_id"].iloc[0] if "park_id" in df.columns and len(df) else f.stem.replace("park","")

        df = df.drop_duplicates(subset=["t"]).set_index("t").sort_index()

        g = pd.DataFrame(index=full)
        g["park_id"] = pid
        g["pm25"] = df["pm25"].reindex(full)
        g["is_missing"] = g["pm25"].isna().astype(int)

        out = OUT_DIR / f.name
        g.reset_index(names="t").to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] {f.name}: missing={g['is_missing'].sum():,} hours -> {out}")

if __name__ == "__main__":
    main()
