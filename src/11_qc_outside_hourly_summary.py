#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDIR = ROOT / "analysis" / "data_processed" / "outside_hourly"
OUT = ROOT / "analysis" / "outputs" / "qc_outside_hourly_summary.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    rows = []
    for f in sorted(INDIR.glob("outside_*.csv")):
        df = pd.read_csv(f)
        if "time" not in df.columns or "pm25" not in df.columns or len(df) == 0:
            rows.append([f.name, 0, "", "", np.nan, np.nan, np.nan])
            continue

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        n = len(df)
        tmin = df["time"].min()
        tmax = df["time"].max()
        pm = pd.to_numeric(df["pm25"], errors="coerce")

        n_na = int(pm.isna().sum())
        pct_na = float(n_na / n) if n else np.nan
        n_neg = int((pm < 0).sum())

        rows.append([f.name, n, tmin, tmax, n_na, pct_na, n_neg])

    outdf = pd.DataFrame(rows, columns=[
        "file","n_rows","time_min","time_max","pm25_na","pm25_na_frac","pm25_negative_count"
    ])
    outdf.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {OUT}")
    print(outdf.sort_values("n_rows").head(10).to_string(index=False))

if __name__ == "__main__":
    main()
