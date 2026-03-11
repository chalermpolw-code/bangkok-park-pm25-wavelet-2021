# analysis/src/38_make_landuse_two_panel_2021_publication.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from adjustText import adjust_text

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
    rho, p = spearmanr(x[m], y[m])
    return float(rho), float(p), int(m.sum())

def short_label(pair_id: str) -> str:
    # "park16_outside28_2021" -> "P16"
    p = pair_id.split("_")[0]  # "park16"
    num = "".join(ch for ch in p if ch.isdigit())
    return f"P{num}" if num else p

def main():
    df = pd.read_csv(INFILE)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))

    for ax, (tag, buf, xcol, ycol, xlabel, ylabel) in zip(axes, PANELS):
        d = df[df["buffer_m"] == buf].copy()
        x = d[xcol].to_numpy(float)
        y = d[ycol].to_numpy(float)
        rho, p, n = sp(x, y)

        sns.regplot(
            x=x, y=y, ax=ax,
            scatter_kws={"s": 70, "alpha": 0.85},
            line_kws={"color": "gray", "linestyle": "--", "linewidth": 1.5},
            ci=None
        )

        # Reference lines: y=0 dB and Δ=0
        ax.axhline(0, linestyle=":", linewidth=1.0, color="gray", alpha=0.6)
        ax.axvline(0, linestyle=":", linewidth=1.0, color="gray", alpha=0.6)

        # Give a little breathing room for labels near plot edges
        ax.margins(0.05)

        texts = []
        for _, row in d.iterrows():
            lab = short_label(str(row["pair_id"]))
            texts.append(ax.text(row[xcol], row[ycol], lab, fontsize=9))

        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.7)
        )

        ax.set_xlabel(xlabel, fontweight="bold", labelpad=10)
        ax.set_ylabel(ylabel, fontweight="bold", labelpad=10)
        ax.set_title(f"({tag}) buffer={buf} m | Spearman ρ={rho:.2f} (n={n})", pad=15)

    fig.tight_layout()
    out_pdf = os.path.join(OUTDIR, "Fig_landuse_two_panel_2021_publication.pdf")
    out_png = os.path.join(OUTDIR, "Fig_landuse_two_panel_2021_publication.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[OK] wrote", out_pdf)
    print("[OK] wrote", out_png)

if __name__ == "__main__":
    main()