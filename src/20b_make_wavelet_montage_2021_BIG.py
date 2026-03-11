#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
20b_make_wavelet_montage_2021_BIG.py

Creates BIG, readable montages (4×2 panels) from existing PNGs:
  - attenuation_matrix
  - coherence_matrix
  - power_inside
  - power_outside

Auto-crops extra whitespace around each PNG to enlarge readable content.

Inputs:
  analysis/outputs/wavelet_2021/figures/
    parkXX_outsideYY_2021_attenuation_matrix.png
    parkXX_outsideYY_2021_coherence_matrix.png
    parkXX_outsideYY_2021_power_inside.png
    parkXX_outsideYY_2021_power_outside.png

Outputs:
  analysis/outputs/wavelet_2021/figures/
    montageBIG_2021_attenuation_matrix.pdf (+ .png)
    montageBIG_2021_coherence_matrix.pdf (+ .png)
    montageBIG_2021_power_inside.pdf (+ .png)
    montageBIG_2021_power_outside.pdf (+ .png)
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

try:
    from PIL import Image, ImageChops
    PIL_OK = True
except Exception:
    PIL_OK = False

FIG_DIR = "analysis/outputs/wavelet_2021/figures"
OUT_PREFIX = os.path.join(FIG_DIR, "montageBIG_2021")

# 4×2 gives larger, readable panels
NROWS, NCOLS = 4, 2

# BIG figure size (inches) — feel free to adjust bigger
FIGSIZE = (20, 26)

# High DPI export for PNG
DPI = 300

# Font sizes
SUPTITLE_FS = 22
TITLE_FS = 14

PATTERN = re.compile(
    r"(park\d{2}_outside\d{2}_2021)_(attenuation_matrix|coherence_matrix|power_inside|power_outside)\.png$"
)

def list_pairs(fig_dir: str):
    pairs = set()
    for fp in glob.glob(os.path.join(fig_dir, "*.png")):
        m = PATTERN.search(os.path.basename(fp))
        if m:
            pairs.add(m.group(1))

    def key(p):
        m = re.match(r"park(\d{2})_outside(\d{2})_2021", p)
        return (int(m.group(1)), int(m.group(2)))

    return sorted(pairs, key=key)

def autocrop_pil(img: Image.Image, border=10) -> Image.Image:
    """
    Crop whitespace border around an image (best-effort).
    """
    # Convert to RGB for safe operations
    im = img.convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return im
    left, upper, right, lower = bbox
    left = max(0, left - border)
    upper = max(0, upper - border)
    right = min(im.size[0], right + border)
    lower = min(im.size[1], lower + border)
    return im.crop((left, upper, right, lower))

def read_image(fp: str):
    if PIL_OK:
        img = Image.open(fp)
        img = autocrop_pil(img, border=6)
        return np.array(img)
    else:
        # fallback (no cropping)
        return plt.imread(fp)

def make_grid(pairs, kind: str, suptitle: str, out_pdf: str, out_png: str):
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(pairs):
            continue

        pair = pairs[i]
        fp = os.path.join(FIG_DIR, f"{pair}_{kind}.png")
        if not os.path.exists(fp):
            ax.set_title(f"{pair}\n(MISSING)", fontsize=TITLE_FS)
            continue

        img = read_image(fp)
        ax.imshow(img)
        ax.set_title(pair, fontsize=TITLE_FS, pad=8)

    fig.suptitle(suptitle, fontsize=SUPTITLE_FS, y=0.995)

    # Tight layout with room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

    print(f"[DONE] wrote {out_pdf}")
    print(f"[DONE] wrote {out_png}")

def main():
    pairs = list_pairs(FIG_DIR)
    if not pairs:
        raise SystemExit(f"No matching PNGs found in {FIG_DIR}")

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