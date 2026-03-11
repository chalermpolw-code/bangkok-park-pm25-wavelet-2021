#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derive annulus / ring-gradient water features from cumulative buffer means.

Input
-----
analysis/data_processed/landuse_2021/pairs_landuse_water_wavelet_2021_ALLBUFFERS.csv

Output
------
analysis/data_processed/landuse_2021/pairs_water_annulus_wavelet_2021.csv
"""

import os
import numpy as np
import pandas as pd

IN_CSV = "analysis/data_processed/landuse_2021/pairs_landuse_water_wavelet_2021_ALLBUFFERS.csv"
OUT_CSV = "analysis/data_processed/landuse_2021/pairs_water_annulus_wavelet_2021.csv"


def weighted_annulus_mean(mean_outer, n_outer, mean_inner, n_inner):
    if pd.isna(mean_outer) or pd.isna(n_outer):
        return np.nan
    if mean_inner is None or n_inner is None:
        return mean_outer
    if pd.isna(mean_inner) or pd.isna(n_inner):
        return np.nan

    n_ann = n_outer - n_inner
    if n_ann <= 0:
        return np.nan

    sum_outer = mean_outer * n_outer
    sum_inner = mean_inner * n_inner
    return (sum_outer - sum_inner) / n_ann


def build_pair(g: pd.DataFrame) -> dict:
    g = g.sort_values("buffer_m").copy()

    expected_buffers = [250, 500, 1000]
    got = g["buffer_m"].tolist()
    if got != expected_buffers:
        raise ValueError(f"Pair {g['pair_id'].iloc[0]} has unexpected buffers: {got}")

    r250 = g[g["buffer_m"] == 250].iloc[0]
    r500 = g[g["buffer_m"] == 500].iloc[0]
    r1000 = g[g["buffer_m"] == 1000].iloc[0]

    out = {}

    # Metadata
    for c in [
        "pair_id", "park_id", "outside_station_id", "distance_km",
        "park_lat", "park_lon", "outside_lat", "outside_lon",
        "outside_district_th", "outside_site_th", "outside_type_th", "rank"
    ]:
        if c in r250.index:
            out[c] = r250[c]

    # Wavelet regime metrics
    out["att_short"] = np.mean([r250["att_2-6h"], r250["att_6-18h"]])
    out["att_diurnal"] = r250["att_18-30h (diurnal)"]
    out["att_long"] = np.mean([r250["att_2-7d"], r250["att_7-14d"]])

    out["coh_short"] = np.mean([r250["coh_2-6h"], r250["coh_6-18h"]])
    out["coh_diurnal"] = r250["coh_18-30h (diurnal)"]
    out["coh_long"] = np.mean([r250["coh_2-7d"], r250["coh_7-14d"]])

    keep_wave = [
        "att_2-6h", "att_6-18h", "att_18-30h (diurnal)", "att_2-7d", "att_7-14d",
        "coh_2-6h", "coh_6-18h", "coh_18-30h (diurnal)", "coh_2-7d", "coh_7-14d"
    ]
    for c in keep_wave:
        out[c] = r250[c]

    metrics = ["NDWI", "MNDWI", "WATER_FRAC"]

    # Preserve cumulative means
    for prefix in ["park", "out"]:
        for idx in metrics:
            for rr, row in [(250, r250), (500, r500), (1000, r1000)]:
                out[f"{prefix}_{idx}_cum_{rr}"] = row[f"{prefix}_{idx}"]
                out[f"{prefix}_{idx}_cum_{rr}_n"] = row[f"{prefix}_{idx}_n"]

    # Annulus means
    for prefix in ["park", "out"]:
        for idx in metrics:
            out[f"{prefix}_{idx}_ring_0_250"] = weighted_annulus_mean(
                r250[f"{prefix}_{idx}"], r250[f"{prefix}_{idx}_n"], None, None
            )
            out[f"{prefix}_{idx}_ring_250_500"] = weighted_annulus_mean(
                r500[f"{prefix}_{idx}"], r500[f"{prefix}_{idx}_n"],
                r250[f"{prefix}_{idx}"], r250[f"{prefix}_{idx}_n"]
            )
            out[f"{prefix}_{idx}_ring_500_1000"] = weighted_annulus_mean(
                r1000[f"{prefix}_{idx}"], r1000[f"{prefix}_{idx}_n"],
                r500[f"{prefix}_{idx}"], r500[f"{prefix}_{idx}_n"]
            )

    # Ring counts
    for prefix in ["park", "out"]:
        for idx in metrics:
            out[f"{prefix}_{idx}_ring_0_250_n"] = r250[f"{prefix}_{idx}_n"]
            out[f"{prefix}_{idx}_ring_250_500_n"] = r500[f"{prefix}_{idx}_n"] - r250[f"{prefix}_{idx}_n"]
            out[f"{prefix}_{idx}_ring_500_1000_n"] = r1000[f"{prefix}_{idx}_n"] - r500[f"{prefix}_{idx}_n"]

    # Park - outside contrasts by ring
    for idx in metrics:
        for ring in ["0_250", "250_500", "500_1000"]:
            out[f"d{idx}_ring_{ring}"] = out[f"park_{idx}_ring_{ring}"] - out[f"out_{idx}_ring_{ring}"]

    # Radial gradients within each side
    for prefix in ["park", "out"]:
        for idx in metrics:
            out[f"{prefix}_{idx}_core_minus_mid"] = out[f"{prefix}_{idx}_ring_0_250"] - out[f"{prefix}_{idx}_ring_250_500"]
            out[f"{prefix}_{idx}_core_minus_outer"] = out[f"{prefix}_{idx}_ring_0_250"] - out[f"{prefix}_{idx}_ring_500_1000"]
            out[f"{prefix}_{idx}_mid_minus_outer"] = out[f"{prefix}_{idx}_ring_250_500"] - out[f"{prefix}_{idx}_ring_500_1000"]

    # Contrast persistence
    for idx in metrics:
        out[f"d{idx}_inner_minus_outer"] = out[f"d{idx}_ring_0_250"] - out[f"d{idx}_ring_500_1000"]
        out[f"d{idx}_mid_minus_outer"] = out[f"d{idx}_ring_250_500"] - out[f"d{idx}_ring_500_1000"]
        out[f"{idx}_contrast_persistence"] = (
            np.mean([out[f"d{idx}_ring_0_250"], out[f"d{idx}_ring_250_500"]]) - out[f"d{idx}_ring_500_1000"]
        )

    # Intuitive summaries
    out["wet_core_advantage"] = out["dMNDWI_ring_0_250"]
    out["wet_mid_advantage"] = out["dMNDWI_ring_250_500"]
    out["wet_outer_advantage"] = out["dMNDWI_ring_500_1000"]

    out["surface_water_core_advantage"] = out["dWATER_FRAC_ring_0_250"]
    out["surface_water_mid_advantage"] = out["dWATER_FRAC_ring_250_500"]
    out["surface_water_outer_advantage"] = out["dWATER_FRAC_ring_500_1000"]

    return out


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    df = pd.read_csv(IN_CSV)

    required = {
        "pair_id", "buffer_m",
        "park_NDWI", "park_NDWI_n", "park_MNDWI", "park_MNDWI_n", "park_WATER_FRAC", "park_WATER_FRAC_n",
        "out_NDWI", "out_NDWI_n", "out_MNDWI", "out_MNDWI_n", "out_WATER_FRAC", "out_WATER_FRAC_n",
        "att_2-6h", "att_6-18h", "att_18-30h (diurnal)", "att_2-7d", "att_7-14d",
        "coh_2-6h", "coh_6-18h", "coh_18-30h (diurnal)", "coh_2-7d", "coh_7-14d",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for pair_id, g in df.groupby("pair_id"):
        rows.append(build_pair(g))

    out = pd.DataFrame(rows)

    first_cols = [
        "pair_id", "park_id", "outside_station_id", "distance_km",
        "att_short", "att_diurnal", "att_long",
        "coh_short", "coh_diurnal", "coh_long",
        "wet_core_advantage", "wet_mid_advantage", "wet_outer_advantage",
        "surface_water_core_advantage", "surface_water_mid_advantage", "surface_water_outer_advantage",
        "MNDWI_contrast_persistence", "WATER_FRAC_contrast_persistence",
    ]
    keep = [c for c in first_cols if c in out.columns] + [c for c in out.columns if c not in first_cols]
    out = out[keep]

    out.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {OUT_CSV}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()