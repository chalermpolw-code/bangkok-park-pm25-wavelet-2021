#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, re
import numpy as np
import matplotlib.pyplot as plt

ARR_DIR = "analysis/data_processed/wavelet_arrays_2021"
OUT_DIR = "analysis/outputs/wavelet_2021/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Layout: High-visibility landscape formatting
NROWS, NCOLS = 2, 4
FIGSIZE = (22, 10) # Wide landscape configuration

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

PAIRKEY = re.compile(r"(park\d{2}_outside\d{2}_2021)\.npz$")

def load_all():
    fps = sorted(glob.glob(os.path.join(ARR_DIR, "*.npz")))
    items = []
    for fp in fps:
        m = PAIRKEY.search(os.path.basename(fp))
        if not m:
            continue
        data = np.load(fp, allow_pickle=True)
        pair = str(data["pair"])
        items.append((pair, dict(data)))

    def key(p):
        m = re.match(r"park(\d{2})_outside(\d{2})_2021", p[0])
        return (int(m.group(1)), int(m.group(2)))
    items.sort(key=key)
    return items

def global_limits(mats, p_lo=1, p_hi=99):
    vals = np.concatenate([m[np.isfinite(m)].ravel() for m in mats if np.isfinite(m).any()])
    if vals.size == 0:
        return (-1, 1)
    return (np.percentile(vals, p_lo), np.percentile(vals, p_hi))

def plot_grid(kind, title, getter, cmap, vmin, vmax, outname, cbar_label):
    items = load_all()

    # Constrained layout works perfectly for wide grids as well
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, sharex=True, sharey=True, constrained_layout=True)
    axes = axes.flatten()

    im0 = None
    periods_h = items[0][1]["periods_h"]

    for i, (pair, d) in enumerate(items):
        ax = axes[i]
        mat = getter(d)

        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        if im0 is None:
            im0 = im

        clean_title = pair.replace("_2021", "").replace("_", " ").title()
        ax.set_title(clean_title, fontweight='bold', pad=10)

        # Draw and shade the Cone of Influence (COI)
        coi_h = d["coi_h"]
        t_indices = np.arange(len(coi_h))

        coi_h_safe = np.clip(coi_h, periods_h[0], None)
        coi_idx = np.interp(np.log(coi_h_safe), np.log(periods_h), np.arange(len(periods_h)))

        ax.plot(t_indices, coi_idx, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.fill_between(t_indices, coi_idx, len(periods_h)-1, color="white", alpha=0.4, zorder=2, hatch='x')

    # Turn off unused axes if you have fewer than 8 plots
    for j in range(len(items), len(axes)):
        axes[j].axis("off")

    # Y-ticks (Periods in Hours)
    yticks = np.linspace(0, len(periods_h)-1, 8)
    ylabels = [f"{periods_h[int(k)]:.1f}" for k in yticks]

    for ax in axes[:len(items)]:
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

    # Clean outer axis labels dynamically
    for i, ax in enumerate(axes[:len(items)]):
        if i % NCOLS == 0: # Far left column
            ax.set_ylabel("Period (Hours)", fontweight='bold')
        if i >= len(items) - NCOLS: # Bottom row
            ax.set_xlabel("Time (Hours)", fontweight='bold')

    fig.suptitle(title, fontsize=18, fontweight='bold')

    # Shared colorbar
    cbar = fig.colorbar(im0, ax=axes.tolist(), shrink=0.85, aspect=30, pad=0.02)
    cbar.set_label(cbar_label, fontsize=16, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    out_pdf = os.path.join(OUT_DIR, outname + ".pdf")
    fig.savefig(out_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[DONE] wrote {out_pdf}")

def main():
    items = load_all()
    if not items:
        raise SystemExit("No .npz wavelet arrays found. Run 21_export_wavelet_arrays_2021.py first.")

    att_mats = [d["att_db"] for _, d in items]
    coh_mats = [d["coh_r2"] for _, d in items]
    pin_mats = [np.log10(d["pin"] + 1e-12) for _, d in items]
    pout_mats= [np.log10(d["pout"] + 1e-12) for _, d in items]

    # Global limits
    att_lo, att_hi = global_limits(att_mats, 1, 99)
    M = max(abs(att_lo), abs(att_hi))
    att_lo, att_hi = -M, M # Symmetric

    coh_lo, coh_hi = 0.0, 1.0
    pin_lo, pin_hi = global_limits(pin_mats, 1, 99)
    pout_lo, pout_hi = global_limits(pout_mats, 1, 99)
    p_lo = min(pin_lo, pout_lo)
    p_hi = max(pin_hi, pout_hi)

    # 1. Attenuation (RdBu_r colormap)
    plot_grid(
        "att", "2021 Wavelet Attenuation",
        getter=lambda d: d["att_db"],
        cmap="RdBu_r", vmin=att_lo, vmax=att_hi,
        outname="montage_SHARED_2021_attenuation",
        cbar_label="Attenuation (dB)"
    )

    # 2. Coherence
    plot_grid(
        "coh", "2021 Wavelet Coherence",
        getter=lambda d: d["coh_r2"],
        cmap="viridis", vmin=coh_lo, vmax=coh_hi,
        outname="montage_SHARED_2021_coherence",
        cbar_label="Coherence ($R^2$)"
    )

    # 3. Power Inside
    plot_grid(
        "pin", "2021 Wavelet Power (Inside)",
        getter=lambda d: np.log10(d["pin"] + 1e-12),
        cmap="viridis", vmin=p_lo, vmax=p_hi,
        outname="montage_SHARED_2021_power_inside",
        cbar_label="Log10(Power)"
    )

    # 4. Power Outside
    plot_grid(
        "pout", "2021 Wavelet Power (Outside)",
        getter=lambda d: np.log10(d["pout"] + 1e-12),
        cmap="viridis", vmin=p_lo, vmax=p_hi,
        outname="montage_SHARED_2021_power_outside",
        cbar_label="Log10(Power)"
    )

if __name__ == "__main__":
    main()