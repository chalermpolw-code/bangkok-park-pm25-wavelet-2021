#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spike audit for cleaned park PM2.5 series.

Reads:
  analysis/data_processed/parks_hourly_clean/parkXX.csv

Writes:
  analysis/outputs/spike_audit_top20_per_park.csv
  analysis/outputs/spike_audit_summary.csv

Behavior:
- For each park, list top N highest pm25 hours (timestamp + value).
- Also summarize how many points exceed common thresholds (e.g., 150, 200, 300, 500).
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "analysis" / "data_processed" / "parks_hourly_clean"
OUT_DIR = ROOT / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TZ = "Asia/Bangkok"
TOP_N = 20
THRESHOLDS = [150, 200, 300, 500]

def main():
    all_top = []
    summary = []

    for f in sorted(IN_DIR.glob("park*.csv")):
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(TZ)
        df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

        pid = df["park_id"].iloc[0] if "park_id" in df.columns and len(df) else f.stem.replace("park", "")
        d = df.dropna(subset=["pm25"]).copy()

        # top-N
        top = d.nlargest(TOP_N, "pm25")[["t", "pm25"]].copy()
        top.insert(0, "park_id", pid)
        top.insert(1, "park_file", f.name)
        all_top.append(top)

        # threshold counts
        row = {"park_id": pid, "park_file": f.name, "n_valid": len(d)}
        for thr in THRESHOLDS:
            row[f"n_ge_{thr}"] = int((d["pm25"] >= thr).sum())
        row["pm25_max"] = float(d["pm25"].max()) if len(d) else None
        row["t_at_max"] = d.loc[d["pm25"].idxmax(), "t"] if len(d) else None
        summary.append(row)

        print(f"[OK] {f.name}: max={row['pm25_max']}")

    top_out = OUT_DIR / "spike_audit_top20_per_park.csv"
    summ_out = OUT_DIR / "spike_audit_summary.csv"

    pd.concat(all_top, ignore_index=True).to_csv(top_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).sort_values("pm25_max", ascending=False).to_csv(summ_out, index=False, encoding="utf-8-sig")

    print(f"[DONE] wrote {top_out}")
    print(f"[DONE] wrote {summ_out}")

if __name__ == "__main__":
    main()

