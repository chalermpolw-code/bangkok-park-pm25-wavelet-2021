# analysis/src/32_plot_landuse_vs_wavelet_2021.py
# Simple first-pass test: dNDVI/dNDBI vs attenuation/coherence bandmeans (n=8)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

INFILE = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021.csv"
OUTDIR = "analysis/outputs/landuse_2021"
os.makedirs(OUTDIR, exist_ok=True)

# Choose which band to test first (diurnal is a strong starting point)
BAND = "18-30h (diurnal)"

X_VARS = ["dNDVI", "dNDBI"]
Y_VARS = [f"att_{BAND}", f"coh_{BAND}"]  # attenuation and coherence for same band

def spearman_summary(x, y):
    # drop NaNs pairwise
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())

def make_scatter(df, xcol, ycol, outfile):
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    r, p, n = spearman_summary(x, y)

    plt.figure()
    plt.scatter(x, y)
    for _, row in df.iterrows():
        # label points by pair_id (short)
        pid = str(row["pair_id"]).replace("_2021", "")
        plt.annotate(pid, (row[xcol], row[ycol]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{xcol} vs {ycol} | Spearman r={r:.2f}, p={p:.3g}, n={n}")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def main():
    df = pd.read_csv(INFILE)

    # Save stats table
    stats_rows = []
    for xcol in X_VARS:
        for ycol in Y_VARS:
            r, p, n = spearman_summary(df[xcol].to_numpy(float), df[ycol].to_numpy(float))
            stats_rows.append({"x": xcol, "y": ycol, "spearman_r": r, "p_value": p, "n": n, "band": BAND})

            outfig = os.path.join(OUTDIR, f"scatter_{xcol}_vs_{ycol}.png".replace(" ", ""))
            make_scatter(df, xcol, ycol, outfig)

    stats = pd.DataFrame(stats_rows)
    stats_out = os.path.join(OUTDIR, "stats_landuse_vs_wavelet_2021.csv")
    stats.to_csv(stats_out, index=False)

    print(f"[OK] wrote {stats_out}")
    print(f"[OK] wrote {len(stats_rows)} scatter PNGs to {OUTDIR}")

if __name__ == "__main__":
    main()