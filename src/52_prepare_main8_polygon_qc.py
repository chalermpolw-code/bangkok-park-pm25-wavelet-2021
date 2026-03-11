#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a QC package for the 8 main wavelet parks.

Inputs
------
- analysis/data_processed/morphology_2021/park_osm_candidates.geojson
- analysis/data_processed/morphology_2021/park_selected_polygons.geojson
- analysis/outputs/morphology_2021/park_osm_match_summary.csv
- analysis/data_raw/sites_parks_20_bkk_pm25.csv

Outputs
-------
- analysis/data_processed/morphology_2021/qc_main8/park_main8_candidates.geojson
- analysis/data_processed/morphology_2021/qc_main8/park_main8_points.geojson
- analysis/outputs/morphology_2021/park_main8_candidate_summary.csv
- analysis/outputs/morphology_2021/park_main8_qc_table.csv
- analysis/outputs/morphology_2021/park_main8_qc_map.html   (optional, if folium installed)

How to use
----------
1) Run this script
2) Open park_main8_candidate_summary.csv and park_main8_qc_map.html
3) Edit park_main8_qc_table.csv and fill chosen_rank for any park that needs fixing
   - leave blank to keep the default top-ranked candidate (rank=1)
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

MAIN8 = [3, 9, 11, 12, 13, 16, 18, 20]

PARKS_CSV = "analysis/data_raw/sites_parks_20_bkk_pm25.csv"
IN_CANDIDATES = "analysis/data_processed/morphology_2021/park_osm_candidates.geojson"
IN_SELECTED = "analysis/data_processed/morphology_2021/park_selected_polygons.geojson"
IN_SUMMARY = "analysis/outputs/morphology_2021/park_osm_match_summary.csv"

OUTDIR_DATA = "analysis/data_processed/morphology_2021/qc_main8"
OUTDIR_OUT = "analysis/outputs/morphology_2021"

OUT_CAND_GEOJSON = os.path.join(OUTDIR_DATA, "park_main8_candidates.geojson")
OUT_POINTS_GEOJSON = os.path.join(OUTDIR_DATA, "park_main8_points.geojson")
OUT_CAND_SUMMARY = os.path.join(OUTDIR_OUT, "park_main8_candidate_summary.csv")
OUT_QC_TABLE = os.path.join(OUTDIR_OUT, "park_main8_qc_table.csv")
OUT_MAP_HTML = os.path.join(OUTDIR_OUT, "park_main8_qc_map.html")

LAT_COL = "latitude"
LON_COL = "longitude"
PARK_ID_COL = "park_id"
PARK_NAME_CANDIDATES = ["park_name_th", "park_name_en", "park_name", "name"]


def pick_name_col(df):
    for c in PARK_NAME_CANDIDATES:
        if c in df.columns:
            return c
    return None


def main():
    os.makedirs(OUTDIR_DATA, exist_ok=True)
    os.makedirs(OUTDIR_OUT, exist_ok=True)

    parks = pd.read_csv(PARKS_CSV)
    cand = gpd.read_file(IN_CANDIDATES)
    selected = gpd.read_file(IN_SELECTED)
    summary = pd.read_csv(IN_SUMMARY)

    name_col = pick_name_col(parks)

    # Candidate subset
    cand8 = cand[cand["park_id"].isin(MAIN8)].copy()
    cand8.to_file(OUT_CAND_GEOJSON, driver="GeoJSON")

    # Candidate summary CSV
    keep_cols = [
        "park_id", "park_name", "candidate_rank_local", "osm_name",
        "contains_point", "centroid_dist_m", "boundary_dist_m",
        "name_similarity", "score", "area_m2_candidate",
        "leisure", "landuse", "boundary", "name", "name:th", "official_name"
    ]
    keep_cols = [c for c in keep_cols if c in cand8.columns]
    cand8_df = pd.DataFrame(cand8.drop(columns="geometry"))[keep_cols].copy()
    cand8_df = cand8_df.sort_values(["park_id", "candidate_rank_local"])
    cand8_df.to_csv(OUT_CAND_SUMMARY, index=False)

    # Park points GeoJSON
    parks8 = parks[parks[PARK_ID_COL].isin(MAIN8)].copy()
    points = []
    for _, r in parks8.iterrows():
        points.append({
            "park_id": int(r[PARK_ID_COL]),
            "park_name": str(r[name_col]) if name_col is not None else "",
            "geometry": Point(float(r[LON_COL]), float(r[LAT_COL])),
        })
    gdf_points = gpd.GeoDataFrame(points, geometry="geometry", crs="EPSG:4326")
    gdf_points.to_file(OUT_POINTS_GEOJSON, driver="GeoJSON")

    # QC table to edit manually
    sum8 = summary[summary["park_id"].isin(MAIN8)].copy()
    sum8 = sum8.sort_values("park_id")

    def suggestion(row):
        pid = int(row["park_id"])
        if pid == 18:
            return "FIX_REQUIRED"
        if row.get("selected_contains_point", 0) == 0:
            return "CHECK"
        return "LIKELY_OK"

    qc_rows = []
    for _, r in sum8.iterrows():
        qc_rows.append({
            "park_id": int(r["park_id"]),
            "park_name": r.get("park_name", ""),
            "current_status": r.get("status", ""),
            "current_selected_name": r.get("selected_name", ""),
            "current_selected_area_m2": r.get("selected_area_m2", None),
            "suggested_action": suggestion(r),
            "chosen_rank": "",  # <-- user edits this
            "notes": "",
        })

    qc = pd.DataFrame(qc_rows).sort_values("park_id")
    qc.to_csv(OUT_QC_TABLE, index=False)

    print(f"[OK] wrote {OUT_CAND_GEOJSON}")
    print(f"[OK] wrote {OUT_POINTS_GEOJSON}")
    print(f"[OK] wrote {OUT_CAND_SUMMARY}")
    print(f"[OK] wrote {OUT_QC_TABLE}")

    # Optional Folium map
    try:
        import folium

        center_lat = parks8[LAT_COL].mean()
        center_lon = parks8[LON_COL].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

        # Add candidate polygons
        for _, row in cand8.to_crs("EPSG:4326").iterrows():
            popup = (
                f"park_id={row.get('park_id')}<br>"
                f"candidate_rank={row.get('candidate_rank_local')}<br>"
                f"osm_name={row.get('osm_name','')}<br>"
                f"contains_point={row.get('contains_point','')}<br>"
                f"score={row.get('score','')}<br>"
                f"area_m2={row.get('area_m2_candidate','')}"
            )
            folium.GeoJson(
                row.geometry.__geo_interface__,
                tooltip=popup,
                popup=popup,
                style_function=lambda x: {
                    "color": "#3388ff",
                    "weight": 2,
                    "fillOpacity": 0.10
                }
            ).add_to(m)

        # Add currently selected polygons
        sel8 = selected[selected["park_id"].isin(MAIN8)].copy().to_crs("EPSG:4326")
        for _, row in sel8.iterrows():
            popup = (
                f"CURRENT SELECTED<br>"
                f"park_id={row.get('park_id')}<br>"
                f"selected_name={row.get('selected_name','')}<br>"
                f"score={row.get('selected_score','')}<br>"
                f"contains_point={row.get('selected_contains_point','')}"
            )
            folium.GeoJson(
                row.geometry.__geo_interface__,
                tooltip=popup,
                popup=popup,
                style_function=lambda x: {
                    "color": "#d62728",
                    "weight": 3,
                    "fillOpacity": 0.05
                }
            ).add_to(m)

        # Add station points
        for _, row in gdf_points.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color="black",
                fill=True,
                fill_opacity=1.0,
                popup=f"park_id={row['park_id']}<br>{row['park_name']}"
            ).add_to(m)

        m.save(OUT_MAP_HTML)
        print(f"[OK] wrote {OUT_MAP_HTML}")
    except Exception as e:
        print(f"[WARN] Could not write folium map: {e}")

    print()
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()