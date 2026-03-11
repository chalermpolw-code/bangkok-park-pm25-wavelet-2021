#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan annulus / ring-gradient features against wavelet regime metrics.

Input
-----
analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ANNULUS.csv

What this script does
---------------------
1) Selects a curated set of interpretable annulus / gradient features
2) Tests them against wavelet regime metrics using Spearman correlation
3) Writes a ranked summary table
4) Prints the strongest relations to console

Important
---------
n is very small (8 pairs), so treat this as exploratory effect-size scanning.
Do NOT overclaim p-values.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IN_CSV = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ANNULUS.csv"
OUT_CSV = "analysis/outputs/landuse_2021/stats_annulus_vs_wavelet_2021.csv"


def safe_spearman(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    rho, p = spearmanr(x[ok], y[ok])
    return float(rho), float(p), int(ok.sum())


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    df = pd.read_csv(IN_CSV)

    wavelet_targets = [
        "att_short", "att_diurnal", "att_long",
        "coh_short", "coh_diurnal", "coh_long",
        "att_2-6h", "att_6-18h", "att_18-30h (diurnal)", "att_2-7d", "att_7-14d",
        "coh_2-6h", "coh_6-18h", "coh_18-30h (diurnal)", "coh_2-7d", "coh_7-14d",
    ]

    candidate_features = [
        # ring contrasts
        "dNDVI_ring_0_250", "dNDVI_ring_250_500", "dNDVI_ring_500_1000",
        "dNDBI_ring_0_250", "dNDBI_ring_250_500", "dNDBI_ring_500_1000",

        # persistence / gradients
        "dNDVI_inner_minus_outer", "dNDVI_mid_minus_outer",
        "dNDBI_inner_minus_outer", "dNDBI_mid_minus_outer",
        "green_contrast_persistence", "built_contrast_persistence",

        # intuitive summaries
        "green_core_advantage", "green_mid_advantage", "green_outer_advantage",
        "built_core_advantage", "built_mid_advantage", "built_outer_advantage",

        # internal radial gradients
        "park_NDVI_core_minus_outer", "park_NDBI_core_minus_outer",
        "out_NDVI_core_minus_outer", "out_NDBI_core_minus_outer",

        # geometry
        "distance_km",
    ]

    candidate_features = [c for c in candidate_features if c in df.columns]
    wavelet_targets = [c for c in wavelet_targets if c in df.columns]

    rows = []
    for feat in candidate_features:
        for target in wavelet_targets:
            rho, p, n = safe_spearman(df[feat], df[target])
            rows.append({
                "feature": feat,
                "target": target,
                "spearman_rho": rho,
                "p_value": p,
                "n": n,
                "abs_rho": np.abs(rho) if pd.notna(rho) else np.nan,
            })

    out = pd.DataFrame(rows).sort_values(["abs_rho", "feature", "target"], ascending=[False, True, True])
    out.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote {OUT_CSV}")
    print("\n=== TOP 25 RELATIONS BY |rho| ===")
    print(out.head(25).to_string(index=False))

    print("\n=== TOP RELATIONS FOR REGIME METRICS ONLY ===")
    regime_only = out[out["target"].isin(["att_short", "att_diurnal", "att_long", "coh_short", "coh_diurnal", "coh_long"])].copy()
    print(regime_only.head(25).to_string(index=False))


if __name__ == "__main__":
    main()