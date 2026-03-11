#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute morphology / shape metrics for selected park polygons.

Input
-----
analysis/data_processed/morphology_2021/park_selected_polygons.geojson

Output
------
analysis/data_processed/morphology_2021/park_morphology_metrics_2021.csv
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

IN_GEOJSON = "analysis/data_processed/morphology_2021/park_selected_polygons.geojson"
OUT_CSV = "analysis/data_processed/morphology_2021/park_morphology_metrics_2021.csv"

TARGET_CRS_METRIC = "EPSG:32647"  # Bangkok


def min_rotated_rect_sides(geom):
    """
    Return major/minor side lengths of the minimum rotated rectangle.
    """
    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return np.nan, np.nan

    # 4 unique sides from closed rectangle
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


def compute_metrics(row):
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

    core_frac_25 = float(core_area_25_m2 / area_m2)
    core_frac_50 = float(core_area_50_m2 / area_m2)
    core_frac_100 = float(core_area_100_m2 / area_m2)

    edge_area_25_m2 = float(area_m2 - core_area_25_m2)
    edge_area_50_m2 = float(area_m2 - core_area_50_m2)
    edge_area_100_m2 = float(area_m2 - core_area_100_m2)

    edge_frac_25 = float(edge_area_25_m2 / area_m2)
    edge_frac_50 = float(edge_area_50_m2 / area_m2)
    edge_frac_100 = float(edge_area_100_m2 / area_m2)

    out = {
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
        "core_frac_25": core_frac_25,
        "core_frac_50": core_frac_50,
        "core_frac_100": core_frac_100,

        "edge_area_25_m2": edge_area_25_m2,
        "edge_area_50_m2": edge_area_50_m2,
        "edge_area_100_m2": edge_area_100_m2,
        "edge_frac_25": edge_frac_25,
        "edge_frac_50": edge_frac_50,
        "edge_frac_100": edge_frac_100,
    }
    return out


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    gdf = gpd.read_file(IN_GEOJSON)
    if len(gdf) == 0:
        raise ValueError("Selected polygon file is empty.")

    gdf = gdf.to_crs(TARGET_CRS_METRIC)

    rows = []
    for _, row in gdf.iterrows():
        out = compute_metrics(row)
        if out is not None:
            rows.append(out)

    df = pd.DataFrame(rows).sort_values("park_id")
    df.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote {OUT_CSV}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()