import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

INFILE = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv"
OUTDIR = "analysis/outputs/landuse_2021/paper_figures"
os.makedirs(OUTDIR, exist_ok=True)

PANELS = [
    # (panel_tag, buffer, x, y, xlabel, ylabel)
    ("A", 250, "dNDBI", "att_6-18h",
     "ΔNDBI (park − outside) within 250 m", "Attenuation (dB), 6–18 h band"),
    ("B", 500, "dNDVI", "att_7-14d",
     "ΔNDVI (park − outside) within 500 m", "Attenuation (dB), 7–14 d band"),
]

def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())

def main():
    df = pd.read_csv(INFILE)

    for tag, buf, xcol, ycol, xlabel, ylabel in PANELS:
        d = df[df["buffer_m"] == buf].copy()
        x = d[xcol].to_numpy(float)
        y = d[ycol].to_numpy(float)
        r, p, n = sp(x, y)

        plt.figure(figsize=(6.0, 4.6))
        plt.scatter(x, y)

        # label points lightly (optional but useful with n=8)
        for _, row in d.iterrows():
            pid = str(row["pair_id"]).replace("_2021", "")
            plt.annotate(pid, (row[xcol], row[ycol]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Panel {tag} | buffer={buf} m | Spearman ρ={r:.2f} (n={n})")
        plt.tight_layout()

        png = os.path.join(OUTDIR, f"Fig_landuse_panel_{tag}_2021.png")
        pdf = os.path.join(OUTDIR, f"Fig_landuse_panel_{tag}_2021.pdf")
        plt.savefig(png, dpi=300)
        plt.savefig(pdf)
        plt.close()

        print("[OK] wrote", png)
        print("[OK] wrote", pdf)

if __name__ == "__main__":
    main()