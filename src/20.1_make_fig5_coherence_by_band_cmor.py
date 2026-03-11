from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIG
# ============================================================

PROJECT_ROOT = Path("/Users/chalermpolw/Desktop/raw_data/urban_rhythm_damping_green_parks_bkk_pm25_wavelet_regimes 2")

COH_PATH = PROJECT_ROOT / "analysis" / "outputs" / "wavelet_2021_cmor_final" / "coherence_bandmeans.csv"
OUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "Fig5_coherence_by_band_2021.pdf"
OUT_PNG = OUT_DIR / "Fig5_coherence_by_band_2021.png"


# ============================================================
# SETTINGS & STYLE
# ============================================================

BANDS = ["2-6h", "6-18h", "18-30h (diurnal)", "2-7d", "7-14d"]
BAND_LABELS = ["2–6 h", "6–18 h", "18–30 h\n(diurnal)", "2–7 d", "7–14 d"]

FIGSIZE = (10.5, 6)
DPI = 300

# Professional Q1 Color Palette (Matching Figure 4)
COLOR_BOX_FACE = "#f1f3f5"
COLOR_BOX_EDGE = "#495057"
COLOR_MEDIAN = "#d95f02"
COLOR_POINTS = "#2b83ba"
COLOR_ZERO = "#212529"


def melt_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["pair"] + [b for b in BANDS if b in df.columns]
    out = df[cols].copy()
    out = out.melt(id_vars=["pair"], var_name="band", value_name="coherence_r2")
    out["band"] = pd.Categorical(out["band"], categories=BANDS, ordered=True)
    return out.sort_values(["band", "pair"]).reset_index(drop=True)


def main() -> None:
    coh = pd.read_csv(COH_PATH)
    coh_long = melt_long(coh)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    data = [
        coh_long.loc[coh_long["band"] == b, "coherence_r2"].dropna().values
        for b in BANDS
    ]

    # Q1-styled Boxplot
    ax.boxplot(
        data,
        tick_labels=BAND_LABELS,
        showfliers=False,
        widths=0.55,
        patch_artist=True,
        boxprops=dict(facecolor=COLOR_BOX_FACE, color=COLOR_BOX_EDGE, linewidth=1.2),
        whiskerprops=dict(color=COLOR_BOX_EDGE, linewidth=1.2),
        capprops=dict(color=COLOR_BOX_EDGE, linewidth=1.2),
        medianprops=dict(color=COLOR_MEDIAN, linewidth=2.5),
        zorder=1
    )

    # Deterministic small jitter for individual parks
    for i, b in enumerate(BANDS, start=1):
        sub = coh_long.loc[coh_long["band"] == b, ["pair", "coherence_r2"]].dropna()
        n = len(sub)
        if n == 0:
            continue
        xs = [i + (j - (n - 1) / 2) * 0.04 for j in range(n)]

        # Q1-styled Scatter Points
        ax.plot(
            xs,
            sub["coherence_r2"].values,
            marker="o",
            markerfacecolor=COLOR_POINTS,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linestyle="None",
            markersize=7,
            alpha=0.85,
            zorder=2
        )

    # Styling axes and background
    ax.set_ylim(0, 1.02) # Bounded between 0 and 1 (with a tiny padding at top)
    ax.set_ylabel(r"Band-mean wavelet coherence ($R^2$)", fontsize=12, labelpad=10)
    ax.set_xlabel("Wavelet band", fontsize=12, labelpad=10)

    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

    # Add ticks to all 4 sides and point them inward for a classic scientific look
    ax.tick_params(axis="both", labelsize=11, direction="in", top=True, right=True, length=5, width=1.2)

    # Keep all 4 spines (frame) and set uniform thickness
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color(COLOR_ZERO)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {OUT_PDF}")
    print(f"[OK] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()