import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

INFILE = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv"
OUTDIR = "analysis/outputs/landuse_2021/paper_figures"
os.makedirs(OUTDIR, exist_ok=True)

PANELS = [
    ("A", 250, "dNDBI", "att_6-18h",
     "ΔNDBI (park − outside) within 250 m", "Attenuation (dB), 6–18 h"),
    ("B", 500, "dNDVI", "att_7-14d",
     "ΔNDVI (park − outside) within 500 m", "Attenuation (dB), 7–14 d"),
]

def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())

def main():
    df = pd.read_csv(INFILE)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for ax, (tag, buf, xcol, ycol, xlabel, ylabel) in zip(axes, PANELS):
        d = df[df["buffer_m"] == buf].copy()
        x = d[xcol].to_numpy(float)
        y = d[ycol].to_numpy(float)
        r, p, n = sp(x, y)

        ax.scatter(x, y)
        for _, row in d.iterrows():
            pid = str(row["pair_id"]).replace("_2021", "")
            ax.annotate(pid, (row[xcol], row[ycol]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({tag}) buffer={buf} m | Spearman ρ={r:.2f} (n={n})")

    fig.tight_layout()
    out_pdf = os.path.join(OUTDIR, "Fig_landuse_two_panel_2021.pdf")
    out_png = os.path.join(OUTDIR, "Fig_landuse_two_panel_2021.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print("[OK] wrote", out_pdf)
    print("[OK] wrote", out_png)

if __name__ == "__main__":
    main()