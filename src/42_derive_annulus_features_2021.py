#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derive annulus / ring-gradient features from cumulative buffer means.

Input
-----
analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv

Expected structure
------------------
One row per pair per cumulative buffer (250, 500, 1000 m), with columns such as:
- pair_id
- buffer_m
- park_NDVI, park_NDVI_n
- park_NDBI, park_NDBI_n
- out_NDVI,  out_NDVI_n
- out_NDBI,  out_NDBI_n
- wavelet columns:
    att_2-6h, att_6-18h, att_18-30h (diurnal), att_2-7d, att_7-14d
    coh_2-6h, coh_6-18h, coh_18-30h (diurnal), coh_2-7d, coh_7-14d

What this script does
---------------------
1) Converts cumulative means (0-250, 0-500, 0-1000) into true annulus means:
   - ring_0_250
   - ring_250_500
   - ring_500_1000
2) Computes park-outside contrasts within each ring
3) Computes ring-gradient / persistence summary features
4) Computes wavelet regime summary metrics:
   - att_short
   - att_diurnal
   - att_long
   - coh_short
   - coh_diurnal
   - coh_long
5) Writes one row per pair with all annulus features

Outputs
-------
- analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ANNULUS.csv
"""

import os
import numpy as np
import pandas as pd

IN_CSV = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv"
OUT_CSV = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ANNULUS.csv"


def weighted_annulus_mean(mean_outer, n_outer, mean_inner, n_inner):
    """
    Recover annulus mean from cumulative means:
        annulus_mean = (outer_sum - inner_sum) / (outer_n - inner_n)
    where outer_sum = mean_outer * n_outer
    """
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


def build_annulus_features_for_pair(g: pd.DataFrame) -> dict:
    g = g.sort_values("buffer_m").copy()

    expected_buffers = [250, 500, 1000]
    got = g["buffer_m"].tolist()
    if got != expected_buffers:
        raise ValueError(f"Pair {g['pair_id'].iloc[0]} has unexpected buffers: {got}")

    r250 = g[g["buffer_m"] == 250].iloc[0]
    r500 = g[g["buffer_m"] == 500].iloc[0]
    r1000 = g[g["buffer_m"] == 1000].iloc[0]

    out = {}

    # Keep identifiers / metadata from one row
    keep_meta = [
        "pair_id",
        "park_id",
        "outside_station_id",
        "park_lat",
        "park_lon",
        "outside_lat",
        "outside_lon",
        "distance_km",
        "outside_district_th",
        "outside_site_th",
        "outside_type_th",
        "rank",
    ]
    for c in keep_meta:
        if c in r250.index:
            out[c] = r250[c]

    # -------- Wavelet summary metrics --------
    out["att_short"] = np.mean([r250["att_2-6h"], r250["att_6-18h"]])
    out["att_diurnal"] = r250["att_18-30h (diurnal)"]
    out["att_long"] = np.mean([r250["att_2-7d"], r250["att_7-14d"]])

    out["coh_short"] = np.mean([r250["coh_2-6h"], r250["coh_6-18h"]])
    out["coh_diurnal"] = r250["coh_18-30h (diurnal)"]
    out["coh_long"] = np.mean([r250["coh_2-7d"], r250["coh_7-14d"]])

    # Also preserve original bandwise wavelet values
    keep_wavelet = [
        "att_2-6h", "att_6-18h", "att_18-30h (diurnal)", "att_2-7d", "att_7-14d",
        "coh_2-6h", "coh_6-18h", "coh_18-30h (diurnal)", "coh_2-7d", "coh_7-14d"
    ]
    for c in keep_wavelet:
        out[c] = r250[c]

    # -------- Cumulative means for reference --------
    for prefix in ["park", "out"]:
        for idx in ["NDVI", "NDBI"]:
            for rr, row in [(250, r250), (500, r500), (1000, r1000)]:
                out[f"{prefix}_{idx}_cum_{rr}"] = row[f"{prefix}_{idx}"]
                out[f"{prefix}_{idx}_cum_{rr}_n"] = row[f"{prefix}_{idx}_n"]

    # -------- True annulus means --------
    for prefix in ["park", "out"]:
        for idx in ["NDVI", "NDBI"]:
            # 0-250
            out[f"{prefix}_{idx}_ring_0_250"] = weighted_annulus_mean(
                r250[f"{prefix}_{idx}"], r250[f"{prefix}_{idx}_n"],
                None, None
            )

            # 250-500
            out[f"{prefix}_{idx}_ring_250_500"] = weighted_annulus_mean(
                r500[f"{prefix}_{idx}"], r500[f"{prefix}_{idx}_n"],
                r250[f"{prefix}_{idx}"], r250[f"{prefix}_{idx}_n"]
            )

            # 500-1000
            out[f"{prefix}_{idx}_ring_500_1000"] = weighted_annulus_mean(
                r1000[f"{prefix}_{idx}"], r1000[f"{prefix}_{idx}_n"],
                r500[f"{prefix}_{idx}"], r500[f"{prefix}_{idx}_n"]
            )

    # -------- Ring counts --------
    for prefix in ["park", "out"]:
        for idx in ["NDVI", "NDBI"]:
            out[f"{prefix}_{idx}_ring_0_250_n"] = r250[f"{prefix}_{idx}_n"]
            out[f"{prefix}_{idx}_ring_250_500_n"] = r500[f"{prefix}_{idx}_n"] - r250[f"{prefix}_{idx}_n"]
            out[f"{prefix}_{idx}_ring_500_1000_n"] = r1000[f"{prefix}_{idx}_n"] - r500[f"{prefix}_{idx}_n"]

    # -------- Park - outside contrasts by ring --------
    for idx in ["NDVI", "NDBI"]:
        for ring in ["0_250", "250_500", "500_1000"]:
            out[f"d{idx}_ring_{ring}"] = (
                out[f"park_{idx}_ring_{ring}"] - out[f"out_{idx}_ring_{ring}"]
            )

    # -------- Radial gradients within park and outside --------
    for prefix in ["park", "out"]:
        for idx in ["NDVI", "NDBI"]:
            out[f"{prefix}_{idx}_core_minus_mid"] = (
                out[f"{prefix}_{idx}_ring_0_250"] - out[f"{prefix}_{idx}_ring_250_500"]
            )
            out[f"{prefix}_{idx}_core_minus_outer"] = (
                out[f"{prefix}_{idx}_ring_0_250"] - out[f"{prefix}_{idx}_ring_500_1000"]
            )
            out[f"{prefix}_{idx}_mid_minus_outer"] = (
                out[f"{prefix}_{idx}_ring_250_500"] - out[f"{prefix}_{idx}_ring_500_1000"]
            )

    # -------- Contrast persistence / edge-pressure summaries --------
    # NDVI contrast persistence: does green advantage stay strong outward?
    out["dNDVI_inner_minus_outer"] = out["dNDVI_ring_0_250"] - out["dNDVI_ring_500_1000"]
    out["dNDVI_mid_minus_outer"] = out["dNDVI_ring_250_500"] - out["dNDVI_ring_500_1000"]

    # NDBI contrast persistence / built-edge pressure
    out["dNDBI_inner_minus_outer"] = out["dNDBI_ring_0_250"] - out["dNDBI_ring_500_1000"]
    out["dNDBI_mid_minus_outer"] = out["dNDBI_ring_250_500"] - out["dNDBI_ring_500_1000"]

    # Core protection ideas
    out["green_core_advantage"] = out["dNDVI_ring_0_250"]
    out["green_mid_advantage"] = out["dNDVI_ring_250_500"]
    out["green_outer_advantage"] = out["dNDVI_ring_500_1000"]

    out["built_core_advantage"] = -out["dNDBI_ring_0_250"]   # higher = park less built than outside
    out["built_mid_advantage"] = -out["dNDBI_ring_250_500"]
    out["built_outer_advantage"] = -out["dNDBI_ring_500_1000"]

    # Persistence as mean inner/mid minus outer
    out["green_contrast_persistence"] = np.mean([
        out["dNDVI_ring_0_250"], out["dNDVI_ring_250_500"]
    ]) - out["dNDVI_ring_500_1000"]

    out["built_contrast_persistence"] = -(
        np.mean([out["dNDBI_ring_0_250"], out["dNDBI_ring_250_500"]]) - out["dNDBI_ring_500_1000"]
    )

    return out


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    df = pd.read_csv(IN_CSV)

    required = {
        "pair_id", "buffer_m",
        "park_NDVI", "park_NDVI_n", "park_NDBI", "park_NDBI_n",
        "out_NDVI", "out_NDVI_n", "out_NDBI", "out_NDBI_n",
        "att_2-6h", "att_6-18h", "att_18-30h (diurnal)", "att_2-7d", "att_7-14d",
        "coh_2-6h", "coh_6-18h", "coh_18-30h (diurnal)", "coh_2-7d", "coh_7-14d",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for pair_id, g in df.groupby("pair_id"):
        rows.append(build_annulus_features_for_pair(g))

    out = pd.DataFrame(rows)

    # Useful ordering
    first_cols = [
        "pair_id", "park_id", "outside_station_id", "distance_km",
        "att_short", "att_diurnal", "att_long",
        "coh_short", "coh_diurnal", "coh_long",
        "green_core_advantage", "green_mid_advantage", "green_outer_advantage",
        "built_core_advantage", "built_mid_advantage", "built_outer_advantage",
        "green_contrast_persistence", "built_contrast_persistence",
    ]
    keep = [c for c in first_cols if c in out.columns] + [c for c in out.columns if c not in first_cols]
    out = out[keep]

    out.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote {OUT_CSV}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()