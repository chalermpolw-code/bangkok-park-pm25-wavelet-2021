from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# TYPOGRAPHY & STYLE
# ============================================================
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# ============================================================
# PATHS
# ============================================================
HARDCODED_ROOT = Path("/Users/chalermpolw/Desktop/raw_data/urban_rhythm_damping_green_parks_bkk_pm25_wavelet_regimes 2")
PROJECT_ROOT = Path.cwd() if Path("analysis/outputs").exists() else HARDCODED_ROOT

INFILE = PROJECT_ROOT / "analysis" / "outputs" / "landuse_2021" / "loo_spearman_summary.csv"
OUTDIR = PROJECT_ROOT / "manuscript" / "figures"

OUT_PDF = OUTDIR / "Fig9_loo_robustness_mean_buffer_relationships_2021.pdf"
OUT_PNG = OUTDIR / "Fig9_loo_robustness_mean_buffer_relationships_2021.png"

# ============================================================
# CONFIGURATION
# ============================================================
# Titles updated for publication formatting: letter on top, metric below
TARGETS = [
    (250, "dNDBI", "att_6-18h", "(a)\nΔNDBI 250 m vs 6–18 h attenuation"),
    (500, "dNDBI", "att_7-14d", "(b)\nΔNDBI 500 m vs 7–14 d attenuation"),
    (500, "dNDVI", "att_7-14d", "(c)\nΔNDVI 500 m vs 7–14 d attenuation"),
]

COLOR_RANGE = "#9ecae1"
COLOR_FULL = "#08519c"
COLOR_ZERO = "#212529"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def plot_robustness_bar(ax: plt.Axes, sub_df: pd.DataFrame, title: str) -> None:
    """Plots a single LOO robustness range bar onto the provided axis."""

    if sub_df.empty:
        ax.set_title(f"{title}\n(No Data)", color="red")
        return

    full_sample = sub_df[sub_df["dropped"] == "NONE"]
    loo_samples = sub_df[sub_df["dropped"] != "NONE"]

    full_r = float(full_sample.iloc[0]["r"])
    loo_min = float(loo_samples["r"].min())
    loo_max = float(loo_samples["r"].max())

    # Range bar
    ax.plot([0, 0], [loo_min, loo_max], color=COLOR_RANGE, linewidth=10,
            solid_capstyle="round", zorder=1)

    # End caps
    cap_hw = 0.08
    ax.plot([-cap_hw, cap_hw], [loo_min, loo_min], color=COLOR_ZERO,
            linewidth=1.5, zorder=2)
    ax.plot([-cap_hw, cap_hw], [loo_max, loo_max], color=COLOR_ZERO,
            linewidth=1.5, zorder=2)

    # Full-sample rho
    ax.scatter([0], [full_r], s=95, color=COLOR_FULL, zorder=3)

    # Zero line
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, zorder=0)

    # Axis formatting
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([])

    # Pad ensures the two-line title sits comfortably above the plot
    ax.set_title(title, pad=16)

    # --- Annotations ---
    ax.annotate(f"Full: {full_r:.3f}",
                xy=(0.0, full_r), xytext=(16, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=13)

    ax.annotate(f"LOO max: {loo_max:.3f}",
                xy=(0.0, loo_max), xytext=(0, 10),
                textcoords="offset points", va="bottom", ha="center",
                fontsize=12)

    ax.annotate(f"LOO min: {loo_min:.3f}",
                xy=(0.0, loo_min), xytext=(0, -10),
                textcoords="offset points", va="top", ha="center",
                fontsize=12)

    # Grid and spines
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(COLOR_ZERO)


def main() -> None:
    if not INFILE.exists():
        print(f"Error: Missing input file at:\n{INFILE}", file=sys.stderr)
        sys.exit(1)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INFILE)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6),
                             sharey=True,
                             constrained_layout=True)

    axes[0].set_ylabel("Spearman correlation (ρ)")

    for ax, (buf, xvar, yvar, title) in zip(axes, TARGETS):
        sub = df[(df["buffer_m"] == buf) &
                 (df["x"] == xvar) &
                 (df["y"] == yvar)].copy()
        plot_robustness_bar(ax, sub, title)

    # Save outputs
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote to {OUT_PDF}")
    print(f"[OK] Wrote to {OUT_PNG}")

if __name__ == "__main__":
    main()