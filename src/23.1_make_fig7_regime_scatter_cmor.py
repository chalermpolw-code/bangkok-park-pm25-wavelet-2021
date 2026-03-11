from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# ============================================================
# PATH CONFIG
# ============================================================

PROJECT_ROOT = Path("/Users/chalermpolw/Desktop/raw_data/urban_rhythm_damping_green_parks_bkk_pm25_wavelet_regimes 2")

ATT_SUMMARY = PROJECT_ROOT / "analysis" / "outputs" / "wavelet_2021" / "wavelet_summary.csv"
COH_SUMMARY = PROJECT_ROOT / "analysis" / "outputs" / "wavelet_2021_cmor_final" / "coherence_summary.csv"

OUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "Fig7_regime_scatter_mean_att_mean_coh.pdf"
OUT_PNG = OUT_DIR / "Fig7_regime_scatter_mean_att_mean_coh.png"


# ============================================================
# SETTINGS & STYLE
# ============================================================

FIGSIZE = (10, 7)
DPI = 300
COLOR_ZERO = "#212529"

PARK_NAME_MAP = {
    "park03_outside18_2021": "Chatuchak Park",
    "park09_outside26_2021": "Phra Nakhon Park",
    "park11_outside09_2021": "Lumpini Park",
    "park12_outside18_2021": "Wachirabenchathat Park",
    "park13_outside18_2021": "Queen Sirikit Park",
    "park16_outside28_2021": "Nong Chok Park",
    "park18_outside32_2021": "Suan Luang Rama IX Park",
    "park20_outside30_2021": "Seri Thai Park",
}

REGIME_MAP = {
    "park16_outside28_2021": "Strong decoupler",
    "park18_outside32_2021": "Strong decoupler",
    "park03_outside18_2021": "Broadband damper",
    "park12_outside18_2021": "Broadband damper",
    "park13_outside18_2021": "Broadband damper",
    "park09_outside26_2021": "Weak follower",
    "park11_outside09_2021": "Weak follower",
    "park20_outside30_2021": "Weak follower",
}

# Q1 Professional Color Palette
COLOR_MAP = {
    "Strong decoupler": "#0072B2",    # Deep Blue
    "Broadband damper": "#009E73",    # Rich Teal/Green
    "Weak follower": "#D55E00",       # Rust Orange
}

ELLIPSE_STYLE = {
    "Strong decoupler": dict(xy=(-14.4, 0.515), width=3.8, height=0.09, angle=0),
    "Broadband damper": dict(xy=(-6.75, 0.562), width=1.35, height=0.05, angle=0),
    "Weak follower": dict(xy=(-2.2, 0.665), width=3.4, height=0.12, angle=0),
}

# Manual label offsets and alignments: (dx, dy, horizontalalignment, verticalalignment)
LABEL_OFFSETS = {
    "Nong Chok Park": (0, 0.010, "center", "bottom"),
    "Suan Luang Rama IX Park": (0, -0.010, "center", "top"),
    "Chatuchak Park": (0.65, 0.010, "left", "center"),
    "Wachirabenchathat Park": (0.65, 0.000, "left", "center"),
    "Queen Sirikit Park": (0.65, -0.010, "left", "center"),
    "Phra Nakhon Park": (0.10, 0.010, "left", "bottom"),
    "Lumpini Park": (0.10, -0.010, "left", "top"),
    "Seri Thai Park": (0.10, 0.000, "left", "center"),
}


def load_data() -> pd.DataFrame:
    att = pd.read_csv(ATT_SUMMARY)[["pair", "mean_attenuation_db"]].copy()
    coh = pd.read_csv(COH_SUMMARY)[["pair", "mean_coherence_R2"]].copy()

    df = att.merge(coh, on="pair", how="inner")
    df["park_name"] = df["pair"].map(PARK_NAME_MAP)
    df["regime"] = df["pair"].map(REGIME_MAP)

    missing_names = df["park_name"].isna().sum()
    missing_regimes = df["regime"].isna().sum()
    if missing_names or missing_regimes:
        raise ValueError("Missing park name or regime mapping for one or more pairs.")

    return df


def main() -> None:
    df = load_data()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Regime ellipses first (so they sit behind the points)
    for regime, params in ELLIPSE_STYLE.items():
        e = Ellipse(
            **params,
            facecolor=COLOR_MAP[regime],
            edgecolor=COLOR_MAP[regime],
            linewidth=1.5,
            alpha=0.15,
            zorder=1,
        )
        ax.add_patch(e)

    # Points by regime
    for regime in ["Strong decoupler", "Broadband damper", "Weak follower"]:
        sub = df[df["regime"] == regime]
        ax.scatter(
            sub["mean_attenuation_db"],
            sub["mean_coherence_R2"],
            s=100,
            label=regime,
            color=COLOR_MAP[regime],
            edgecolors="white", # Clean white border around dots
            linewidths=1.2,
            zorder=3,
        )

    # Labels with subtle readable backgrounds
    for _, row in df.iterrows():
        park = row["park_name"]
        regime = row["regime"]
        x = row["mean_attenuation_db"]
        y = row["mean_coherence_R2"]
        dx, dy, ha, va = LABEL_OFFSETS[park]

        bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)

        # Only add pointing lines for the tightly clustered 'Broadband damper' regime
        if regime == "Broadband damper":
            arrow_props = dict(arrowstyle="-", color=COLOR_ZERO, lw=1.0, alpha=0.5)
        else:
            arrow_props = None

        ax.annotate(
            park,
            xy=(x, y),                 # Point line *to* the data coordinate
            xytext=(x + dx, y + dy),   # Position text at the offset
            fontsize=10,
            zorder=4,
            ha=ha,                     # Use precise alignment to hug the dot
            va=va,
            bbox=bbox_props,
            arrowprops=arrow_props
        )

    # Styling Axes
    ax.set_xlabel("Mean attenuation across five wavelet bands (dB)", fontsize=12, labelpad=10)
    ax.set_ylabel(r"Mean coherence across five wavelet bands ($R^2$)", fontsize=12, labelpad=10)

    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)

    # Legend update: frameon=False removes the box!
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)

    ax.set_xlim(-16.5, 0.5)
    ax.set_ylim(0.46, 0.72)

    # Inward ticks and uniform frame thickness
    ax.tick_params(axis="both", labelsize=11, direction="in", top=True, right=True, length=5, width=1.2)
    for spine in ax.spines.values():
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