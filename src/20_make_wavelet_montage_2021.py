#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20_make_wavelet_montage_2021.py

Creates 4 montage figures (8 subplots each) from existing PNGs:
  - attenuation_matrix
  - coherence_matrix
  - power_inside
  - power_outside

Inputs (expected):
  analysis/outputs/wavelet_2021/figures/
    parkXX_outsideYY_2021_attenuation_matrix.png
    parkXX_outsideYY_2021_coherence_matrix.png
    parkXX_outsideYY_2021_power_inside.png
    parkXX_outsideYY_2021_power_outside.png

Outputs:
  analysis/outputs/wavelet_2021/figures/montage_2021_attenuation_matrix.pdf (+ .png)
  analysis/outputs/wavelet_2021/figures/montage_2021_coherence_matrix.pdf (+ .png)
  analysis/outputs/wavelet_2021/figures/montage_2021_power_inside.pdf (+ .png)
  analysis/outputs/wavelet_2021/figures/montage_2021_power_outside.pdf (+ .png)
"""

import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


FIG_DIR = "analysis/outputs/wavelet_2021/figures"
OUT_PREFIX = os.path.join(FIG_DIR, "montage_2021")

# 2 rows x 4 cols = 8 panels
NROWS, NCOLS = 2, 4

# If your images have large margins, you can reduce whitespace:
TIGHT = True

PATTERN = re.compile(r"(park\d{2}_outside\d{2}_2021)_(attenuation_matrix|coherence_matrix|power_inside|power_outside)\.png$")


def list_pairs(fig_dir: str):
    """Discover available pair prefixes like park03_outside18_2021."""
    pairs = set()
    for fp in glob.glob(os.path.join(fig_dir, "*.png")):
        m = PATTERN.search(os.path.basename(fp))
        if m:
            pairs.add(m.group(1))
    # Sort by park number (then outside number)
    def key(p):
        m = re.match(r"park(\d{2})_outside(\d{2})_2021", p)
        return (int(m.group(1)), int(m.group(2)))
    return sorted(pairs, key=key)


def make_grid(pairs, kind: str, title: str, out_pdf: str, out_png: str):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(pairs):
            continue

        pair = pairs[i]
        fp = os.path.join(FIG_DIR, f"{pair}_{kind}.png")
        if not os.path.exists(fp):
            ax.set_title(f"{pair}\n(MISSING)", fontsize=9)
            continue

        img = mpimg.imread(fp)
        ax.imshow(img)
        ax.set_title(pair, fontsize=9)

    fig.suptitle(title, fontsize=14)

    if TIGHT:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[DONE] wrote {out_pdf}")
    print(f"[DONE] wrote {out_png}")


def main():
    pairs = list_pairs(FIG_DIR)
    if len(pairs) == 0:
        raise SystemExit(f"No matching PNGs found in {FIG_DIR}")

    # You expect 8 — but we won't hard-fail if it finds more/less
    print(f"Found {len(pairs)} pair prefixes:")
    for p in pairs:
        print(" -", p)

    make_grid(
        pairs, "attenuation_matrix",
        "2021 Wavelet Attenuation Matrices (8 pairs)",
        f"{OUT_PREFIX}_attenuation_matrix.pdf",
        f"{OUT_PREFIX}_attenuation_matrix.png",
    )

    make_grid(
        pairs, "coherence_matrix",
        "2021 Wavelet Coherence Matrices (8 pairs)",
        f"{OUT_PREFIX}_coherence_matrix.pdf",
        f"{OUT_PREFIX}_coherence_matrix.png",
    )

    make_grid(
        pairs, "power_inside",
        "2021 Wavelet Power (Inside) (8 pairs)",
        f"{OUT_PREFIX}_power_inside.pdf",
        f"{OUT_PREFIX}_power_inside.png",
    )

    make_grid(
        pairs, "power_outside",
        "2021 Wavelet Power (Outside) (8 pairs)",
        f"{OUT_PREFIX}_power_outside.pdf",
        f"{OUT_PREFIX}_power_outside.png",
    )


if __name__ == "__main__":
    main()