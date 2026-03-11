from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.stats import spearmanr

# ============================================================
# TYPOGRAPHY
# ============================================================
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path("/Users/chalermpolw/Desktop/raw_data/urban_rhythm_damping_green_parks_bkk_pm25_wavelet_regimes 2")

INFILE = PROJECT_ROOT / "analysis" / "data_processed" / "landuse_2021" / "pairs_water_annulus_wavelet_2021.csv"
OUTDIR = PROJECT_ROOT / "manuscript" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTDIR / "Fig11_water_context_coherence_relationships_2021.pdf"
OUT_PNG = OUTDIR / "Fig11_water_context_coherence_relationships_2021.png"

# ============================================================
# TARGETS
# ============================================================
TARGETS = [
    (
        "dWATER_FRAC_mid_minus_outer",
        "coh_6-18h",
        "(a) Water-fraction contrast\nvs. 6–18 h coherence",
    ),
    (
        "out_MNDWI_core_minus_outer",
        "coh_2-6h",
        "(b) Outside-site MNDWI core–outer contrast\nvs. 2–6 h coherence",
    ),
]

COLOR_POINT = "#0072B2"
COLOR_ZERO = "#212529"

PAIR_NUM = {
    "park03_outside18_2021": 1,
    "park12_outside18_2021": 2,
    "park13_outside18_2021": 3,
    "park11_outside09_2021": 4,
    "park09_outside26_2021": 5,
    "park16_outside28_2021": 6,
    "park20_outside30_2021": 7,
    "park18_outside32_2021": 8,
}

LABEL_OFFSETS = {
    1: (0.000, 0.020),
    2: (0.000, 0.020),
    3: (0.000, 0.020),
    4: (0.000, 0.020),
    5: (0.000, 0.020),
    6: (0.000, 0.025),
    7: (0.000, 0.020),
    8: (0.000, 0.020),
}

# ============================================================
# HELPERS
# ============================================================

def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    r, p = spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())

def format_p_value(p: float) -> str:
    if p < 0.001:
        return "$p < 0.001$"
    return f"$p = {p:.3f}$"

def pretty_x(xcol: str) -> str:
    mapping = {
        "dWATER_FRAC_mid_minus_outer": r"$\Delta$ water fraction (mid-ring $-$ outer ring)",
        "out_MNDWI_core_minus_outer": "Outside-site MNDWI (core $-$ outer ring)",
        "out_NDWI_core_minus_outer": "Outside-site NDWI (core $-$ outer ring)",
    }
    return mapping.get(xcol, xcol)

def pretty_y(ycol: str) -> str:
    mapping = {
        "coh_2-6h": "2–6 h coherence",
        "coh_6-18h": "6–18 h coherence",
        "coh_short": "Short-band coherence",
    }
    return mapping.get(ycol, ycol)

def main() -> None:
    if not INFILE.exists():
        raise FileNotFoundError(f"Missing point-level file:\n{INFILE}")

    df = pd.read_csv(INFILE)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

    for ax, (xcol, ycol, title) in zip(axes, TARGETS):
        if xcol not in df.columns or ycol not in df.columns:
            raise KeyError(f"Required column not found: {xcol} or {ycol}")

        d = df.copy()
        d["pair_num"] = d["pair_id"].map(PAIR_NUM)
        d = d.sort_values("pair_num")

        x = d[xcol].to_numpy(float)
        y = d[ycol].to_numpy(float)
        r, p, _ = spearman(x, y)

        ax.scatter(
            x,
            y,
            s=80,
            color=COLOR_POINT,
            edgecolor="none",
            linewidth=0,
            zorder=3,
        )

        y_span = float(np.nanmax(y) - np.nanmin(y)) if len(y) else 1.0
        if not np.isfinite(y_span) or y_span <= 0:
            y_span = 1.0

        for _, row in d.iterrows():
            num = int(row["pair_num"])
            dx, dy_mult = LABEL_OFFSETS.get(num, (0.0, 0.020))
            ax.text(
                float(row[xcol]) + dx,
                float(row[ycol]) + dy_mult * y_span,
                f"{num}",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=4,
            )

        p_str = format_p_value(p)
        ax.set_title(
            f"{title}\nSpearman $\\rho = {r:.2f}$, {p_str}",
            fontweight="normal",
            pad=10,
        )

        ax.set_xlabel(pretty_x(xcol), labelpad=6)
        ax.set_ylabel(pretty_y(ycol), labelpad=6)

        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", direction="in", top=True, right=True, length=5, width=1.0)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color(COLOR_ZERO)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {OUT_PDF}")
    print(f"[OK] wrote {OUT_PNG}")

if __name__ == "__main__":
    main()