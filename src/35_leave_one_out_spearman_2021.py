# analysis/src/35_leave_one_out_spearman_2021.py
# Leave-one-out (LOO) Spearman robustness check for selected landuse ↔ attenuation relations (2021)

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INFILE = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv"
OUTDIR = "analysis/outputs/landuse_2021"
os.makedirs(OUTDIR, exist_ok=True)
OUTFILE = os.path.join(OUTDIR, "loo_spearman_summary.csv")

# Targets = the two recommended "first paper figure" relations + a couple of extras (optional)
TARGETS = [
    # (buffer_m, x, y)
    (250, "dNDBI", "att_6-18h"),   # panel A (local, interpretable)
    (500, "dNDVI", "att_7-14d"),   # panel B (long-period context)

    # optional extras (keep or delete if you want only 2)
    (500, "dNDBI", "att_7-14d"),
    (250, "dNDBI", "att_2-6h"),
]

def spearman_xy(x: np.ndarray, y: np.ndarray):
    """Pairwise finite mask, then Spearman."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, n
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), n

def main():
    df = pd.read_csv(INFILE)

    out_rows = []

    for buf, xcol, ycol in TARGETS:
        d = df[df["buffer_m"] == buf].copy()

        # Full-sample
        r_full, p_full, n_full = spearman_xy(d[xcol].to_numpy(float), d[ycol].to_numpy(float))
        out_rows.append({
            "buffer_m": buf, "x": xcol, "y": ycol,
            "dropped": "NONE", "r": r_full, "p": p_full, "n": n_full
        })

        # Leave-one-out
        for pid in sorted(d["pair_id"].unique()):
            dd = d[d["pair_id"] != pid]
            r, p, n = spearman_xy(dd[xcol].to_numpy(float), dd[ycol].to_numpy(float))
            out_rows.append({
                "buffer_m": buf, "x": xcol, "y": ycol,
                "dropped": pid, "r": r, "p": p, "n": n
            })

    out = pd.DataFrame(out_rows)
    out.to_csv(OUTFILE, index=False)
    print(f"[OK] wrote {OUTFILE}")

    # Quick summary printed to console: how much r varies across drops
    loo = out[out["dropped"] != "NONE"].copy()
    if len(loo) > 0:
        summ = (loo.groupby(["buffer_m", "x", "y"])["r"]
                .agg(["min", "max", "mean"])
                .reset_index())
        print("\n[INFO] LOO r-range summary (min/max/mean across dropped pairs):")
        print(summ.to_string(index=False))

if __name__ == "__main__":
    main()