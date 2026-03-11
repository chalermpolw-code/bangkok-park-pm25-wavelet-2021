#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract candidate park polygons from OpenStreetMap around each park station point.

Goal
----
Build a first-pass park footprint layer for morphology analysis.

Logic
-----
For each park station:
1) query OSM features near the station
2) keep polygon-like park/recreation geometries
3) score candidates using:
   - contains station point (strongest)
   - centroid distance
   - optional name similarity
4) select the best candidate polygon

Outputs
-------
- analysis/data_processed/morphology_2021/park_osm_candidates.geojson
- analysis/data_processed/morphology_2021/park_selected_polygons.geojson
- analysis/outputs/morphology_2021/park_osm_match_summary.csv

Notes
-----
- This is intended as a high-quality first pass, not guaranteed perfect.
- If a park polygon is not found or the wrong one is selected, inspect the summary
  and manually fix the selected GeoJSON later if needed.
"""

import os
import math
import difflib
import warnings

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

try:
    import osmnx as ox
except ImportError as e:
    raise ImportError("osmnx is required. Install with: pip install osmnx") from e

# -----------------------------
# CONFIG
# -----------------------------
PARK_SITES_CSV = "analysis/data_raw/sites_parks_20_bkk_pm25.csv"

PARK_ID_COL = "park_id"
LAT_COL = "latitude"
LON_COL = "longitude"
PARK_NAME_COL_CANDIDATES = ["park_name_th", "park_name_en", "park_name", "name"]

SEARCH_RADIUS_M = 700   # broad enough for many urban parks, still local
TOP_N_KEEP = 10

OUTDIR_DATA = "analysis/data_processed/morphology_2021"
OUTDIR_OUT = "analysis/outputs/morphology_2021"

OUT_CANDIDATES = os.path.join(OUTDIR_DATA, "park_osm_candidates.geojson")
OUT_SELECTED = os.path.join(OUTDIR_DATA, "park_selected_polygons.geojson")
OUT_SUMMARY = os.path.join(OUTDIR_OUT, "park_osm_match_summary.csv")

TARGET_CRS_METRIC = "EPSG:32647"  # UTM 47N, appropriate for Bangkok


# -----------------------------
# Helpers
# -----------------------------
def require_columns(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Found: {list(df.columns)}")


def pick_name_col(df: pd.DataFrame):
    for c in PARK_NAME_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def normalize_name(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    for ch in [" ", "_", "-", ".", "/", "(", ")", "[", "]", ",", ";", ":"]:
        s = s.replace(ch, "")
    return s


def name_similarity(a, b):
    a = normalize_name(a)
    b = normalize_name(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def safe_geom_name(row):
    for c in ["name", "name:th", "official_name", "alt_name"]:
        if c in row and pd.notna(row[c]):
            return str(row[c])
    return ""


def is_polygonal(geom: BaseGeometry) -> bool:
    return geom is not None and geom.geom_type in ["Polygon", "MultiPolygon"]


def score_candidate(geom_metric, point_metric, park_name, osm_name):
    """
    Scoring priorities:
    - Contains point: huge bonus
    - Near centroid/boundary: strong
    - Name match: moderate
    """
    contains = geom_metric.contains(point_metric)
    centroid_dist = geom_metric.centroid.distance(point_metric)
    boundary_dist = geom_metric.boundary.distance(point_metric)
    sim = name_similarity(park_name, osm_name)

    score = 0.0
    if contains:
        score += 1000.0

    score += max(0.0, 300.0 - centroid_dist) * 0.5
    score += max(0.0, 200.0 - boundary_dist) * 0.4
    score += sim * 100.0

    area = geom_metric.area
    # Very tiny features are suspicious
    if area < 500:
        score -= 200.0

    return {
        "contains_point": int(bool(contains)),
        "centroid_dist_m": float(centroid_dist),
        "boundary_dist_m": float(boundary_dist),
        "name_similarity": float(sim),
        "score": float(score),
        "area_m2_candidate": float(area),
    }


def query_osm_polygons(lat, lon, radius_m):
    """
    Query OSM features near a point using park-like tags.
    """
    tags = {
        "leisure": ["park", "garden", "recreation_ground", "nature_reserve"],
        "landuse": ["recreation_ground"],
        "boundary": ["protected_area"],
    }

    gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)

    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gdf.reset_index(drop=False)
    if "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.apply(is_polygonal)].copy()
    if len(gdf) == 0:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    return gdf


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTDIR_DATA, exist_ok=True)
    os.makedirs(OUTDIR_OUT, exist_ok=True)

    parks = pd.read_csv(PARK_SITES_CSV)
    require_columns(parks, [PARK_ID_COL, LAT_COL, LON_COL], "parks")
    name_col = pick_name_col(parks)

    all_candidates = []
    selected_rows = []
    summary_rows = []

    for _, r in parks.iterrows():
        park_id = int(r[PARK_ID_COL])
        lat = float(r[LAT_COL])
        lon = float(r[LON_COL])
        park_name = str(r[name_col]) if name_col is not None and pd.notna(r[name_col]) else ""

        print(f"[INFO] park_id={park_id:02d} | querying OSM ...")

        try:
            cand = query_osm_polygons(lat, lon, SEARCH_RADIUS_M)
        except Exception as e:
            summary_rows.append({
                "park_id": park_id,
                "park_name": park_name,
                "status": "OSM_QUERY_FAIL",
                "n_candidates": 0,
                "selected_name": "",
                "selected_score": None,
                "selected_contains_point": None,
                "selected_area_m2": None,
                "error": str(e),
            })
            continue

        if len(cand) == 0:
            summary_rows.append({
                "park_id": park_id,
                "park_name": park_name,
                "status": "NO_CANDIDATES",
                "n_candidates": 0,
                "selected_name": "",
                "selected_score": None,
                "selected_contains_point": None,
                "selected_area_m2": None,
                "error": "",
            })
            continue

        point_gdf = gpd.GeoDataFrame(
            [{"park_id": park_id, "geometry": Point(lon, lat)}],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_crs(TARGET_CRS_METRIC)
        point_metric = point_gdf.geometry.iloc[0]

        cand_metric = cand.to_crs(TARGET_CRS_METRIC).copy()

        scored_rows = []
        for idx, crow in cand_metric.iterrows():
            osm_name = safe_geom_name(crow)
            s = score_candidate(crow.geometry, point_metric, park_name, osm_name)
            row = {
                "park_id": park_id,
                "park_name": park_name,
                "osm_name": osm_name,
                "candidate_rank_local": None,
                "search_radius_m": SEARCH_RADIUS_M,
                **s,
                "geometry": crow.geometry,
            }
            # carry some useful raw tags if present
            for c in ["leisure", "landuse", "boundary", "name", "name:th", "official_name", "osm_id", "element_type"]:
                if c in crow.index:
                    row[c] = crow[c]
            scored_rows.append(row)

        cand_scored = gpd.GeoDataFrame(scored_rows, geometry="geometry", crs=TARGET_CRS_METRIC)
        cand_scored = cand_scored.sort_values(
            ["score", "contains_point", "name_similarity", "area_m2_candidate"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

        cand_scored["candidate_rank_local"] = range(1, len(cand_scored) + 1)

        top_keep = cand_scored.head(TOP_N_KEEP).copy()
        all_candidates.append(top_keep)

        best = cand_scored.iloc[0].copy()

        selected_rows.append({
            "park_id": park_id,
            "park_name": park_name,
            "selected_name": best.get("osm_name", ""),
            "selected_score": best.get("score", None),
            "selected_contains_point": best.get("contains_point", None),
            "selected_area_m2": best.get("area_m2_candidate", None),
            "geometry": best.geometry,
        })

        status = "OK_CONTAINS_POINT" if int(best.get("contains_point", 0)) == 1 else "OK_NEAREST_CANDIDATE"
        summary_rows.append({
            "park_id": park_id,
            "park_name": park_name,
            "status": status,
            "n_candidates": int(len(cand_scored)),
            "selected_name": best.get("osm_name", ""),
            "selected_score": best.get("score", None),
            "selected_contains_point": best.get("contains_point", None),
            "selected_area_m2": best.get("area_m2_candidate", None),
            "error": "",
        })

    # Write outputs
    if len(all_candidates):
        gdf_candidates = pd.concat(all_candidates, ignore_index=True)
        gdf_candidates = gpd.GeoDataFrame(gdf_candidates, geometry="geometry", crs=TARGET_CRS_METRIC).to_crs("EPSG:4326")
        gdf_candidates.to_file(OUT_CANDIDATES, driver="GeoJSON")
        print(f"[OK] wrote {OUT_CANDIDATES}")
    else:
        print("[WARN] No candidate polygons to write.")

    if len(selected_rows):
        gdf_selected = gpd.GeoDataFrame(selected_rows, geometry="geometry", crs=TARGET_CRS_METRIC).to_crs("EPSG:4326")
        gdf_selected.to_file(OUT_SELECTED, driver="GeoJSON")
        print(f"[OK] wrote {OUT_SELECTED}")
    else:
        print("[WARN] No selected polygons to write.")

    df_summary = pd.DataFrame(summary_rows).sort_values("park_id")
    df_summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[OK] wrote {OUT_SUMMARY}")
    print()
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()