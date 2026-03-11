from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

# Thicken the hatch lines slightly for print resolution
mpl.rcParams['hatch.linewidth'] = 0.8


# ============================================================
# PATH CONFIG
# ============================================================

PROJECT_ROOT = Path("/Users/chalermpolw/Desktop/raw_data/urban_rhythm_damping_green_parks_bkk_pm25_wavelet_regimes 2")

NPZ_PATH = PROJECT_ROOT / "analysis" / "data_processed" / "wavelet_arrays_2021" / "park16_outside28_2021.npz"
OUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "Fig6_nongchok_representative_scalograms.pdf"
OUT_PNG = OUT_DIR / "Fig6_nongchok_representative_scalograms.png"


# ============================================================
# SETTINGS & STYLE
# ============================================================

FIGSIZE = (12.5, 5.8)
DPI = 300
EPS = 1e-12

COLOR_ZERO = "#212529"
CMAP = "magma" # Q1 standard for spectral density

BAND_LINES_H = [6, 18, 30, 48, 168]
BAND_LABELS = [
    ("2–6 h", 3.4),
    ("6–18 h", 10.4),
    ("18–30 h", 23.5),
    ("2–7 d", 87),
    ("7–14 d", 242),
]

# Convert day numbers to start-of-month ticks for a 365-day year
MONTH_STARTS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _period_edges(periods_h: np.ndarray) -> np.ndarray:
    mids = np.sqrt(periods_h[:-1] * periods_h[1:])
    first = periods_h[0] ** 2 / mids[0]
    last = periods_h[-1] ** 2 / mids[-1]
    return np.concatenate([[first], mids, [last]])


def _time_edges_days(n_time: int) -> np.ndarray:
    return np.arange(n_time + 1) / 24.0


def _coi_mask_polygon(ax, t_days: np.ndarray, coi_h: np.ndarray, y_top: float) -> None:
    # 1. Dashed boundary line
    ax.plot(t_days, coi_h, linestyle="--", linewidth=1.5, color="white", alpha=0.95)

    # 2. Darkening shade layer (Using RGBA to keep the global alpha at 1.0)
    ax.fill_between(
        t_days,
        coi_h,
        y_top,
        facecolor=(0, 0, 0, 0.65), # 65% opaque black shade
        edgecolor="none",
        linewidth=0.0,
    )

    # 3. Stark white cross-hatching layer on top
    ax.fill_between(
        t_days,
        coi_h,
        y_top,
        facecolor="none",
        edgecolor=(1, 1, 1, 0.6), # 60% opaque white for the cross-hatch
        hatch="xx",               # Dense cross-hatch pattern
        linewidth=0.0,
    )


def _annotate_bands(ax, x_right: float, add_text: bool) -> None:
    for y in BAND_LINES_H:
        ax.axhline(y, linestyle="--", linewidth=0.9, color="white", alpha=0.5)

    if add_text:
        for label, y in BAND_LABELS:
            ax.text(
                x_right,
                y,
                label,
                ha="right",
                va="center",
                fontsize=9.5,
                color=COLOR_ZERO,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
            )


def main() -> None:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing representative pair file: {NPZ_PATH}")

    d = np.load(NPZ_PATH, allow_pickle=True)

    periods_h = d["periods_h"].astype(float)
    pin = d["pin"].astype(float)
    pout = d["pout"].astype(float)
    coi_h = d["coi_h"].astype(float)

    pin_log = np.log10(pin + EPS)
    pout_log = np.log10(pout + EPS)

    vals = np.concatenate([
        pin_log[np.isfinite(pin_log)].ravel(),
        pout_log[np.isfinite(pout_log)].ravel(),
    ])
    vmin = np.percentile(vals, 5)
    vmax = np.percentile(vals, 95)

    n_time = pin.shape[1]
    t_days = np.arange(n_time) / 24.0
    x_edges = _time_edges_days(n_time)
    y_edges = _period_edges(periods_h)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True, constrained_layout=True)

    panels = [
        (axes[0], pin_log, "(a) Park-interior CWT power (Nong Chok Park)"),
        (axes[1], pout_log, "(b) Outside CWT power (paired comparator)"),
    ]

    im0 = None
    for i, (ax, mat, title) in enumerate(panels):
        im = ax.pcolormesh(
            x_edges,
            y_edges,
            mat,
            shading="auto",
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        if im0 is None:
            im0 = im

        ax.set_title(title, fontweight="bold", fontsize=11, pad=12)
        ax.set_yscale("log")
        ax.set_ylim(periods_h[0], periods_h[-1])

        ax.set_xlim(0, 365)
        ax.set_xticks(MONTH_STARTS)
        ax.set_xticklabels(MONTH_NAMES)
        ax.set_xlabel("Time (2021)", fontsize=11)

        _coi_mask_polygon(ax, t_days, coi_h, periods_h[-1])

        _annotate_bands(ax, x_right=t_days[-1] * 0.98, add_text=(i == 1))

        ax.tick_params(axis="both", labelsize=10.5, direction="in", top=True, right=True, length=5, width=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color(COLOR_ZERO)

    axes[0].set_yticks([2, 6, 12, 24, 48, 168, 336])
    axes[0].set_yticklabels(["2h", "6h", "12h", "1d", "2d", "7d", "14d"])
    axes[0].set_ylabel("Wavelet period", fontsize=11)

    cbar = fig.colorbar(im0, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label(r"$\log_{10}(\mathrm{Wavelet\ Power})$", fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=10.5, direction="in", length=4, width=1)
    cbar.outline.set_linewidth(1.2)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {OUT_PDF}")
    print(f"[OK] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()