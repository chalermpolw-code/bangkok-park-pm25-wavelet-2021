import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

INFILE = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv"
OUTDIR = "analysis/outputs/landuse_2021/top_plots"
os.makedirs(OUTDIR, exist_ok=True)

TARGETS = [
    # (buffer_m, x, y)
    (500, "dNDVI", "att_7-14d"),
    (500, "dNDBI", "att_7-14d"),
    (250, "dNDBI", "att_6-18h"),
    (250, "dNDBI", "att_2-6h"),
]

def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())

def plot_one(df, buf, xcol, ycol):
    d = df[df["buffer_m"] == buf].copy()
    x = d[xcol].to_numpy(float)
    y = d[ycol].to_numpy(float)
    r, p, n = sp(x, y)

    plt.figure()
    plt.scatter(x, y)

    for _, row in d.iterrows():
        pid = str(row["pair_id"]).replace("_2021", "")
        plt.annotate(pid, (row[xcol], row[ycol]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    plt.xlabel(f"{xcol} (park - outside)")
    plt.ylabel(ycol)
    plt.title(f"buffer={buf} m | Spearman r={r:.2f}, p={p:.3g}, n={n}")
    plt.tight_layout()

    fname = f"buf{buf}_{xcol}_vs_{ycol}.png".replace(" ", "")
    out = os.path.join(OUTDIR, fname)
    plt.savefig(out, dpi=220)
    plt.close()
    print("[OK] wrote", out)

def main():
    df = pd.read_csv(INFILE)
    for buf, x, y in TARGETS:
        plot_one(df, buf, x, y)

if __name__ == "__main__":
    main()