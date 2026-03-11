import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INFILE = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv"
OUTDIR = "analysis/outputs/landuse_2021"
os.makedirs(OUTDIR, exist_ok=True)
OUTFILE = os.path.join(OUTDIR, "stats_landuse_vs_wavelet_2021_ALLBUFFERS.csv")

BANDS = ["2-6h", "6-18h", "18-30h (diurnal)", "2-7d", "7-14d"]
X_VARS = ["dNDVI", "dNDBI"]

def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())

def main():
    df = pd.read_csv(INFILE)
    rows = []
    for buf in sorted(df["buffer_m"].unique()):
        d = df[df["buffer_m"] == buf]
        for band in BANDS:
            for xcol in X_VARS:
                for kind in ["att", "coh"]:
                    ycol = f"{kind}_{band}"
                    r, p, n = sp(d[xcol].to_numpy(float), d[ycol].to_numpy(float))
                    rows.append({"buffer_m": buf, "band": band, "x": xcol, "y": ycol, "spearman_r": r, "p_value": p, "n": n})
    out = pd.DataFrame(rows)
    out.to_csv(OUTFILE, index=False)
    print(f"[OK] wrote {OUTFILE}")

if __name__ == "__main__":
    main()