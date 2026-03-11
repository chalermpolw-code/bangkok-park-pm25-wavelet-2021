from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

# ============================================================
# TYPOGRAPHY & STYLE
# ============================================================
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ============================================================
# PATHS
# ============================================================
HARDCODED_ROOT = Path("/Users/chalermpolw/Desktop/raw_data/urban_rhythm_damping_green_parks_bkk_pm25_wavelet_regimes 2")
PROJECT_ROOT = Path.cwd() if Path("analysis/outputs").exists() else HARDCODED_ROOT

INFILE = PROJECT_ROOT / "analysis" / "outputs" / "landuse_2021" / "stats_annulus_vs_wavelet_2021.csv"
OUTDIR = PROJECT_ROOT / "manuscript" / "figures"

OUT_PDF = OUTDIR / "Fig10_annulus_ring_gradient_schematic_results_2021.pdf"
OUT_PNG = OUTDIR / "Fig10_annulus_ring_gradient_schematic_results_2021.png"

# ============================================================
# TARGET ROWS
# ============================================================
ROW_ORDER = [
    "Outside NDVI core − outer",
    "Outside NDBI core − outer",
]

COL_ORDER = [
    "2–6 h coherence",
    "6–18 h coherence",
    "18–30 h coherence",
]

TARGET_MAP = {
    ("out_NDVI_core_minus_outer", "coh_2-6h"): ("Outside NDVI core − outer", "2–6 h coherence"),
    ("out_NDVI_core_minus_outer", "coh_6-18h"): ("Outside NDVI core − outer", "6–18 h coherence"),
    ("out_NDVI_core_minus_outer", "coh_18-30h (diurnal)"): ("Outside NDVI core − outer", "18–30 h coherence"),
    ("out_NDBI_core_minus_outer", "coh_2-6h"): ("Outside NDBI core − outer", "2–6 h coherence"),
    ("out_NDBI_core_minus_outer", "coh_6-18h"): ("Outside NDBI core − outer", "6–18 h coherence"),
    ("out_NDBI_core_minus_outer", "coh_18-30h (diurnal)"): ("Outside NDBI core − outer", "18–30 h coherence"),
}

# ============================================================
# HELPERS
# ============================================================
def draw_annulus_schematic(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.axis("off")

    outer = Circle((0, 0), 1.00, facecolor="#f8f9fa", edgecolor="#495057", linewidth=1.5, zorder=1)
    mid = Circle((0, 0), 0.65, facecolor="#e9ecef", edgecolor="#495057", linewidth=1.5, zorder=2)
    core = Circle((0, 0), 0.35, facecolor="#d4edda", edgecolor="#28a745", linewidth=1.5, zorder=3)

    ax.add_patch(outer)
    ax.add_patch(mid)
    ax.add_patch(core)

    # Central outside site
    ax.scatter([0], [0], s=140, color="#08519c", zorder=4)
    ax.text(0, -0.15, "Outside site", ha="center", va="top", fontsize=13, fontweight="bold", color="#08519c")

    # Ring labels vertically centered in their bands
    ax.text(0, 0.16, "Core\n0–250 m", ha="center", va="center", fontsize=12)
    ax.text(0, 0.50, "Mid-ring\n250–500 m", ha="center", va="center", fontsize=12)
    ax.text(0, 0.825, "Outer ring\n500–1000 m", ha="center", va="center", fontsize=12)

    # Feature annotations pulled closer to the circle to eliminate dead space
    ax.text(1.08, 0.45, "Key features:", ha="left", va="center", fontsize=13, fontweight="bold")
    ax.text(1.08, 0.18, "• out_NDVI_core_minus_outer", ha="left", va="center", fontsize=12)
    ax.text(1.08, -0.02, "• out_NDBI_core_minus_outer", ha="left", va="center", fontsize=12)

    # Bottom text pulled up closer to the circle
    ax.text(
        0,
        -1.15,
        "Radial structure around the outside comparator site",
        ha="center",
        va="top",
        fontsize=13
    )

    # Tighter limits effectively "zoom in" the drawing
    ax.set_xlim(-1.05, 2.15)
    ax.set_ylim(-1.30, 1.10)

    # Recalculated exact center over the concentric circles for the tighter xlim
    ax.set_title("(a) Annulus ring-gradient schematic", x=0.328, pad=15)


def build_heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=ROW_ORDER, columns=COL_ORDER, dtype=float)

    for _, row in df.iterrows():
        key = (row["feature"], row["target"])
        if key in TARGET_MAP:
            rlab, clab = TARGET_MAP[key]
            out.loc[rlab, clab] = float(row["spearman_rho"])

    return out


def annotate_heatmap(ax: plt.Axes, mat: pd.DataFrame) -> None:
    # Use white text for dark cells and black text for light cells to maintain contrast
    for i, row_name in enumerate(mat.index):
        for j, col_name in enumerate(mat.columns):
            val = mat.loc[row_name, col_name]
            if pd.notna(val):
                text_color = "white" if abs(val) > 0.45 else "black"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=text_color, fontsize=15)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not INFILE.exists():
        print(f"Error: missing input file:\n{INFILE}", file=sys.stderr)
        sys.exit(1)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INFILE)
    mat = build_heatmap_matrix(df)

    # Shifted width ratio to give panel A more room to grow
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14.5, 5.0),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
        constrained_layout=True
    )

    # Panel A
    draw_annulus_schematic(axes[0])

    # Panel B
    ax = axes[1]

    # Using RdBu_r (Red-Blue reverse)
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-0.8, vmax=0.8)

    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
    annotate_heatmap(ax, mat)

    # Add white gridlines to separate the cells cleanly
    ax.set_xticks(np.arange(-.5, len(mat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title("(b) Strongest annulus–coherence correlations", pad=15)
    ax.set_xlabel("Wavelet coherence band", labelpad=12)

    # Colorbar formatting
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman correlation (ρ)", labelpad=12)
    cbar.outline.set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#343a40")

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {OUT_PDF}")
    print(f"[OK] Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()