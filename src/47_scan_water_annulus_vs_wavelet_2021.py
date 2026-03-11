#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan water-related annulus features against wavelet regime metrics.

Input
-----
analysis/data_processed/landuse_2021/pairs_water_annulus_wavelet_2021.csv

Output
------
analysis/outputs/landuse_2021/stats_water_annulus_vs_wavelet_2021.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IN_CSV = "analysis/data_processed/landuse_2021/pairs_water_annulus_wavelet_2021.csv"
OUT_CSV = "analysis/outputs/landuse_2021/stats_water_annulus_vs_wavelet_2021.csv"


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
    wavelet_targets = [c for c in wavelet_targets if c in df.columns]

    candidate_features = [
        "dNDWI_ring_0_250", "dNDWI_ring_250_500", "dNDWI_ring_500_1000",
        "dMNDWI_ring_0_250", "dMNDWI_ring_250_500", "dMNDWI_ring_500_1000",
        "dWATER_FRAC_ring_0_250", "dWATER_FRAC_ring_250_500", "dWATER_FRAC_ring_500_1000",

        "dNDWI_inner_minus_outer", "dNDWI_mid_minus_outer",
        "dMNDWI_inner_minus_outer", "dMNDWI_mid_minus_outer",
        "dWATER_FRAC_inner_minus_outer", "dWATER_FRAC_mid_minus_outer",

        "NDWI_contrast_persistence", "MNDWI_contrast_persistence", "WATER_FRAC_contrast_persistence",

        "wet_core_advantage", "wet_mid_advantage", "wet_outer_advantage",
        "surface_water_core_advantage", "surface_water_mid_advantage", "surface_water_outer_advantage",

        "park_NDWI_core_minus_outer", "park_MNDWI_core_minus_outer", "park_WATER_FRAC_core_minus_outer",
        "out_NDWI_core_minus_outer", "out_MNDWI_core_minus_outer", "out_WATER_FRAC_core_minus_outer",

        "distance_km",
    ]
    candidate_features = [c for c in candidate_features if c in df.columns]

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
    print("\n=== TOP 25 WATER RELATIONS BY |rho| ===")
    print(out.head(25).to_string(index=False))

    regime_only = out[out["target"].isin(["att_short", "att_diurnal", "att_long", "coh_short", "coh_diurnal", "coh_long"])].copy()
    print("\n=== TOP WATER RELATIONS FOR REGIME METRICS ===")
    print(regime_only.head(25).to_string(index=False))


if __name__ == "__main__":
    main()