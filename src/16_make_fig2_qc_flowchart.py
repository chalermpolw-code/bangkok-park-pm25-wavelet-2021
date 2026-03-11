from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================
# PATH CONFIG
# ============================================================

PROJECT_ROOT = Path("/Users/chalermpolw/Desktop/raw_data/urban_rhythm_damping_green_parks_bkk_pm25_wavelet_regimes 2")
OUTPUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "Fig2_qc_screening_flowchart.png"
OUT_PDF = OUTPUT_DIR / "Fig2_qc_screening_flowchart.pdf"

# ============================================================
# STYLE
# ============================================================

FIGSIZE = (10.5, 5.8)
DPI = 300

COLOR_PARK = "#E8F3EA"
COLOR_OUTSIDE = "#EAF2FB"
COLOR_QC = "#F7F3E8"
COLOR_FINAL = "#F3ECF7"
COLOR_EDGE = "#444444"
TEXT_COLOR = "#222222"
BG_COLOR = "white"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    fontsize: float = 10.5,
    weight: str = "normal",
) -> None:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=COLOR_EDGE,
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=TEXT_COLOR,
        fontweight=weight,
        linespacing=1.15,
        wrap=True,
    )


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.1,
        color=COLOR_EDGE,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def main() -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -----------------------------
    # Row 1: inputs
    # -----------------------------
    add_box(
        ax, 0.10, 0.73, 0.23, 0.16,
        "Park-interior stations\n(n = 20)\n\nHourly PM$_{2.5}$ series for 2021",
        COLOR_PARK,
        fontsize=10.6,
        weight="bold",
    )

    add_box(
        ax, 0.67, 0.73, 0.23, 0.16,
        "Outside stations\n(n = 50)\n\nDistrict-level candidate comparators\nfor 2021",
        COLOR_OUTSIDE,
        fontsize=10.6,
        weight="bold",
    )

    # -----------------------------
    # Row 2: QC
    # -----------------------------
    add_box(
        ax, 0.10, 0.44, 0.23, 0.18,
        "Park preprocessing and QC\n\nStandardization, hourly gridding,\ncleaning, and explicit preservation\nof missingness\n\nAll 20 parks retained",
        COLOR_QC,
        fontsize=9.7,
    )

    add_box(
        ax, 0.67, 0.44, 0.23, 0.18,
        "Outside preprocessing and QC\n\n2021 hourly cleaning and conservative\ncompleteness screen\n(≤ 5\\% missing PM$_{2.5}$)\n\nTier-A outside stations retained\n(n = 6)",
        COLOR_QC,
        fontsize=9.7,
    )

    # -----------------------------
    # Row 3: pairing
    # -----------------------------
    add_box(
        ax, 0.35, 0.13, 0.30, 0.20,
        "Nearest-neighbor geodesic pairing\n\nEach park linked to the nearest retained\nTier-A outside station\n\nFinal retained 2021 park--outside pairs\n(n = 8)\nOverlap = 0.970--0.996 of 8760 h",
        COLOR_FINAL,
        fontsize=10.0,
        weight="bold",
    )

    # -----------------------------
    # Arrows
    # -----------------------------
    add_arrow(ax, 0.215, 0.73, 0.215, 0.62)
    add_arrow(ax, 0.785, 0.73, 0.785, 0.62)

    add_arrow(ax, 0.33, 0.50, 0.44, 0.33)
    add_arrow(ax, 0.67, 0.50, 0.56, 0.33)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

    print(f"[OK] wrote {OUT_PNG}")
    print(f"[OK] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()