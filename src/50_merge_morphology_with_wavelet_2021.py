#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge park morphology metrics with wavelet regime metrics.

Inputs
------
1) analysis/data_processed/morphology_2021/park_morphology_metrics_2021.csv
2) analysis/data_processed/landuse_2021/pairs_water_annulus_wavelet_2021.csv
   (falls back to pairs_landuse_wavelet_2021_ANNULUS.csv if needed)

Output
------
analysis/data_processed/morphology_2021/pairs_morphology_wavelet_2021.csv
"""

import os
import pandas as pd

IN_MORPH = "analysis/data_processed/morphology_2021/park_morphology_metrics_2021.csv"
IN_WAVELET_PREFERRED = "analysis/data_processed/landuse_2021/pairs_water_annulus_wavelet_2021.csv"
IN_WAVELET_FALLBACK = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ANNULUS.csv"
OUT_CSV = "analysis/data_processed/morphology_2021/pairs_morphology_wavelet_2021.csv"


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    morph = pd.read_csv(IN_MORPH)

    if os.path.exists(IN_WAVELET_PREFERRED):
        base = pd.read_csv(IN_WAVELET_PREFERRED)
        print(f"[INFO] using {IN_WAVELET_PREFERRED}")
    else:
        base = pd.read_csv(IN_WAVELET_FALLBACK)
        print(f"[INFO] using {IN_WAVELET_FALLBACK}")

    keep_wave = [
        "pair_id", "park_id", "outside_station_id", "distance_km",
        "att_short", "att_diurnal", "att_long",
        "coh_short", "coh_diurnal", "coh_long",
        "att_2-6h", "att_6-18h", "att_18-30h (diurnal)", "att_2-7d", "att_7-14d",
        "coh_2-6h", "coh_6-18h", "coh_18-30h (diurnal)", "coh_2-7d", "coh_7-14d",
    ]
    keep_wave = [c for c in keep_wave if c in base.columns]
    base = base[keep_wave].drop_duplicates(subset=["pair_id"]).copy()

    morph["park_id"] = morph["park_id"].astype(int)
    base["park_id"] = base["park_id"].astype(int)

    out = base.merge(morph, on="park_id", how="left")
    out.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote {OUT_CSV}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()