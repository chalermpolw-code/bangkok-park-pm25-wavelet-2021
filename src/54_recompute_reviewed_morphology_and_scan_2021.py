#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute morphology metrics and rerun morphology-vs-wavelet scan
using the reviewed polygon layer.

Inputs
------
- analysis/data_processed/morphology_2021/park_selected_polygons_FULL_REVIEWED.geojson
- analysis/data_processed/landuse_2021/pairs_water_annulus_wavelet_2021.csv
  (fallback: pairs_landuse_wavelet_2021_ANNULUS.csv)

Outputs
-------
- analysis/data_processed/morphology_2021/park_morphology_metrics_2021_REVIEWED.csv
- analysis/data_processed/morphology_2021/pairs_morphology_wavelet_2021_REVIEWED.csv
- analysis/outputs/morphology_2021/stats_morphology_vs_wavelet_2021_REVIEWED.csv
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr

IN_GEOJSON = "analysis/data_processed/morphology_2021/park_selected_polygons_FULL_REVIEWED.geojson"
IN_WAVELET_PREFERRED = "analysis/data_processed/landuse_2021/pairs_water_annulus_wavelet_2021.csv"
IN_WAVELET_FALLBACK = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ANNULUS.csv"

OUT_METRICS = "analysis/data_processed/morphology_2021/park_morphology_metrics_2021_REVIEWED.csv"
OUT_MERGED = "analysis/data_processed/morphology_2021/pairs_morphology_wavelet_2021_REVIEWED.csv"
OUT_STATS = "analysis/outputs/morphology_2021/stats_morphology_vs_wavelet_2021_REVIEWED.csv"

TARGET_CRS_METRIC = "EPSG:32647"


def min_rotated_rect_sides(geom):
    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return np.nan, np.nan
    side_lengths = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        side_lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    side_lengths = sorted(side_lengths)
    minor = np.mean(side_lengths[:2])
    major = np.mean(side_lengths[2:])
    return major, minor


def safe_neg_buffer_area(geom, dist_m):
    g = geom.buffer(-dist_m)
    if g.is_empty:
        return 0.0
    return float(g.area)


def compute_metrics_row(row):
    geom = row.geometry
    area_m2 = float(geom.area)
    perimeter_m = float(geom.length)
    convex_area_m2 = float(geom.convex_hull.area)

    if perimeter_m <= 0 or area_m2 <= 0:
        return None

    compactness_polsby = float(4.0 * math.pi * area_m2 / (perimeter_m ** 2))
    perimeter_to_area = float(perimeter_m / area_m2)
    convexity_fill = float(area_m2 / convex_area_m2) if convex_area_m2 > 0 else np.nan

    minx, miny, maxx, maxy = geom.bounds
    bbox_area_m2 = float((maxx - minx) * (maxy - miny))
    bbox_fill_ratio = float(area_m2 / bbox_area_m2) if bbox_area_m2 > 0 else np.nan

    major_axis_m, minor_axis_m = min_rotated_rect_sides(geom)
    elongation_ratio = float(major_axis_m / minor_axis_m) if minor_axis_m and minor_axis_m > 0 else np.nan

    equiv_radius_m = float((area_m2 / math.pi) ** 0.5)
    effective_width_m = float(2.0 * area_m2 / perimeter_m)

    core_area_25_m2 = safe_neg_buffer_area(geom, 25)
    core_area_50_m2 = safe_neg_buffer_area(geom, 50)
    core_area_100_m2 = safe_neg_buffer_area(geom, 100)

    edge_area_25_m2 = float(area_m2 - core_area_25_m2)
    edge_area_50_m2 = float(area_m2 - core_area_50_m2)
    edge_area_100_m2 = float(area_m2 - core_area_100_m2)

    return {
        "park_id": int(row["park_id"]),
        "park_name": row.get("park_name", ""),
        "selected_name": row.get("selected_name", ""),
        "selected_score": row.get("selected_score", np.nan),
        "selected_contains_point": row.get("selected_contains_point", np.nan),

        "area_m2": area_m2,
        "area_ha": area_m2 / 10000.0,
        "perimeter_m": perimeter_m,
        "perimeter_to_area": perimeter_to_area,
        "compactness_polsby": compactness_polsby,
        "convexity_fill": convexity_fill,
        "bbox_area_m2": bbox_area_m2,
        "bbox_fill_ratio": bbox_fill_ratio,
        "major_axis_m": major_axis_m,
        "minor_axis_m": minor_axis_m,
        "elongation_ratio": elongation_ratio,
        "equiv_radius_m": equiv_radius_m,
        "effective_width_m": effective_width_m,

        "core_area_25_m2": core_area_25_m2,
        "core_area_50_m2": core_area_50_m2,
        "core_area_100_m2": core_area_100_m2,
        "core_frac_25": float(core_area_25_m2 / area_m2),
        "core_frac_50": float(core_area_50_m2 / area_m2),
        "core_frac_100": float(core_area_100_m2 / area_m2),

        "edge_area_25_m2": edge_area_25_m2,
        "edge_area_50_m2": edge_area_50_m2,
        "edge_area_100_m2": edge_area_100_m2,
        "edge_frac_25": float(edge_area_25_m2 / area_m2),
        "edge_frac_50": float(edge_area_50_m2 / area_m2),
        "edge_frac_100": float(edge_area_100_m2 / area_m2),
    }


def safe_spearman(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    rho, p = spearmanr(x[ok], y[ok])
    return float(rho), float(p), int(ok.sum())


def main():
    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_STATS), exist_ok=True)

    # Recompute metrics
    gdf = gpd.read_file(IN_GEOJSON).to_crs(TARGET_CRS_METRIC)
    rows = []
    for _, row in gdf.iterrows():
        out = compute_metrics_row(row)
        if out is not None:
            rows.append(out)
    morph = pd.DataFrame(rows).sort_values("park_id")
    morph.to_csv(OUT_METRICS, index=False)
    print(f"[OK] wrote {OUT_METRICS}")

    # Merge with wavelet
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

    base["park_id"] = base["park_id"].astype(int)
    morph["park_id"] = morph["park_id"].astype(int)

    merged = base.merge(morph, on="park_id", how="left")
    merged.to_csv(OUT_MERGED, index=False)
    print(f"[OK] wrote {OUT_MERGED}")

    # Scan
    wavelet_targets = [
        "att_short", "att_diurnal", "att_long",
        "coh_short", "coh_diurnal", "coh_long",
        "att_2-6h", "att_6-18h", "att_18-30h (diurnal)", "att_2-7d", "att_7-14d",
        "coh_2-6h", "coh_6-18h", "coh_18-30h (diurnal)", "coh_2-7d", "coh_7-14d",
    ]
    wavelet_targets = [c for c in wavelet_targets if c in merged.columns]

    candidate_features = [
        "area_ha",
        "perimeter_m",
        "perimeter_to_area",
        "compactness_polsby",
        "convexity_fill",
        "bbox_fill_ratio",
        "elongation_ratio",
        "equiv_radius_m",
        "effective_width_m",
        "core_frac_25",
        "core_frac_50",
        "core_frac_100",
        "edge_frac_25",
        "edge_frac_50",
        "edge_frac_100",
        "distance_km",
    ]
    candidate_features = [c for c in candidate_features if c in merged.columns]

    stats_rows = []
    for feat in candidate_features:
        for target in wavelet_targets:
            rho, p, n = safe_spearman(merged[feat], merged[target])
            stats_rows.append({
                "feature": feat,
                "target": target,
                "spearman_rho": rho,
                "p_value": p,
                "n": n,
                "abs_rho": abs(rho) if pd.notna(rho) else np.nan,
            })

    stats = pd.DataFrame(stats_rows).sort_values(["abs_rho", "feature", "target"], ascending=[False, True, True])
    stats.to_csv(OUT_STATS, index=False)
    print(f"[OK] wrote {OUT_STATS}")

    print("\n=== TOP 30 REVIEWED MORPHOLOGY RELATIONS ===")
    print(stats.head(30).to_string(index=False))


if __name__ == "__main__":
    main()