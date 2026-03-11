#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
19_plot_main_results_2021.py (separate PDFs)

Reads:
  analysis/outputs/wavelet_2021/attenuation_bandmeans.csv
  analysis/outputs/wavelet_2021/coherence_bandmeans.csv

Writes (PDF, separate files):
  analysis/outputs/wavelet_2021/figures/fig1A_attenuation_by_band_2021.pdf
  analysis/outputs/wavelet_2021/figures/fig1B_coherence_by_band_2021.pdf
  analysis/outputs/wavelet_2021/figures/fig1C_scatter_att_vs_coh_2to7d_2021.pdf

Run:
  python analysis/src/19_plot_main_results_2021.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

ATT_PATH = "analysis/outputs/wavelet_2021/attenuation_bandmeans.csv"
COH_PATH = "analysis/outputs/wavelet_2021/coherence_bandmeans.csv"
OUT_DIR = "analysis/outputs/wavelet_2021/figures"

OUT_A = os.path.join(OUT_DIR, "fig1A_attenuation_by_band_2021.pdf")
OUT_B = os.path.join(OUT_DIR, "fig1B_coherence_by_band_2021.pdf")
OUT_C = os.path.join(OUT_DIR, "fig1C_scatter_att_vs_coh_2to7d_2021.pdf")

BANDS = ["2-6h", "6-18h", "18-30h (diurnal)", "2-7d", "7-14d"]
SCAT_BAND = "2-7d"


def melt_long(df, value_name):
    cols = ["pair"] + [b for b in BANDS if b in df.columns]
    df = df[cols].copy()
    long = df.melt(id_vars=["pair"], var_name="band", value_name=value_name)
    long["band"] = pd.Categorical(long["band"], categories=BANDS, ordered=True)
    return long.sort_values(["band", "pair"]).reset_index(drop=True)


def box_with_points(ax, long_df, ycol, ylab, title):
    data = [long_df.loc[long_df["band"] == b, ycol].dropna().values for b in BANDS]

    # Matplotlib 3.9+ prefers tick_labels
    ax.boxplot(data, tick_labels=BANDS, showfliers=False)

    # Overlay points with deterministic small jitter
    for i, b in enumerate(BANDS, start=1):
        sub = long_df[long_df["band"] == b].dropna(subset=[ycol])
        if len(sub) > 0:
            xs = [i + (j - (len(sub)-1)/2)*0.03 for j in range(len(sub))]
            ax.plot(xs, sub[ycol].values, marker="o", linestyle="None", markersize=4)

    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)


def scatter_att_vs_coh(att_df, coh_df, ax):
    a = att_df[["pair", SCAT_BAND]].rename(columns={SCAT_BAND: "att_db"})
    c = coh_df[["pair", SCAT_BAND]].rename(columns={SCAT_BAND: "coh_r2"})
    m = a.merge(c, on="pair", how="inner").dropna(subset=["att_db", "coh_r2"])

    ax.plot(m["att_db"].values, m["coh_r2"].values, marker="o", linestyle="None")

    for _, r in m.iterrows():
        ax.annotate(r["pair"], (r["att_db"], r["coh_r2"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)

    ax.set_xlabel(f"Attenuation (dB) in {SCAT_BAND} (negative = damping)")
    ax.set_ylabel(f"Coherence (R²) in {SCAT_BAND}")
    ax.set_title("Attenuation vs Coherence (2–7 days band)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    ax.set_ylim(0, 1)
    ax.axhline(0.5, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    att = pd.read_csv(ATT_PATH)
    coh = pd.read_csv(COH_PATH)

    att_long = melt_long(att, "attenuation_db")
    coh_long = melt_long(coh, "coherence_r2")

    # Fig 1A: attenuation
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    box_with_points(
        ax,
        att_long,
        "attenuation_db",
        "Attenuation (dB) = 10 log10(P_in / P_out)  (negative = damping)",
        "2021: Attenuation by Timescale Band (Inside vs Outside)"
    )
    fig.tight_layout()
    fig.savefig(OUT_A)
    plt.close(fig)

    # Fig 1B: coherence
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    box_with_points(
        ax,
        coh_long,
        "coherence_r2",
        "Wavelet Coherence (R²)",
        "2021: Coherence by Timescale Band (Inside vs Outside)"
    )
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT_B)
    plt.close(fig)

    # Fig 1C: scatter (2–7d)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    scatter_att_vs_coh(att, coh, ax)
    fig.tight_layout()
    fig.savefig(OUT_C)
    plt.close(fig)

    print("[DONE] wrote:")
    print(" -", OUT_A)
    print(" -", OUT_B)
    print(" -", OUT_C)


if __name__ == "__main__":
    main()