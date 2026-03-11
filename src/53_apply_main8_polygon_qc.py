#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply manual QC choices for the 8 main parks.

Inputs
------
- analysis/data_processed/morphology_2021/qc_main8/park_main8_candidates.geojson
- analysis/outputs/morphology_2021/park_main8_qc_table.csv
- analysis/data_processed/morphology_2021/park_selected_polygons.geojson

Outputs
-------
- analysis/data_processed/morphology_2021/qc_main8/park_selected_polygons_main8_reviewed.geojson
- analysis/data_processed/morphology_2021/park_selected_polygons_FULL_REVIEWED.geojson

Rule
----
If chosen_rank is blank, rank=1 is used.
"""

import os
import pandas as pd
import geopandas as gpd

MAIN8 = [3, 9, 11, 12, 13, 16, 18, 20]

IN_CANDIDATES = "analysis/data_processed/morphology_2021/qc_main8/park_main8_candidates.geojson"
IN_QC_TABLE = "analysis/outputs/morphology_2021/park_main8_qc_table.csv"
IN_ORIG_SELECTED = "analysis/data_processed/morphology_2021/park_selected_polygons.geojson"

OUT_MAIN8_REVIEWED = "analysis/data_processed/morphology_2021/qc_main8/park_selected_polygons_main8_reviewed.geojson"
OUT_FULL_REVIEWED = "analysis/data_processed/morphology_2021/park_selected_polygons_FULL_REVIEWED.geojson"


def main():
    os.makedirs(os.path.dirname(OUT_MAIN8_REVIEWED), exist_ok=True)

    cand = gpd.read_file(IN_CANDIDATES)
    qc = pd.read_csv(IN_QC_TABLE)
    orig = gpd.read_file(IN_ORIG_SELECTED)

    reviewed_rows = []

    for _, r in qc.iterrows():
        park_id = int(r["park_id"])
        chosen_rank = r.get("chosen_rank", "")

        if pd.isna(chosen_rank) or str(chosen_rank).strip() == "":
            chosen_rank = 1
        else:
            chosen_rank = int(float(chosen_rank))

        subset = cand[(cand["park_id"] == park_id) & (cand["candidate_rank_local"] == chosen_rank)].copy()
        if len(subset) == 0:
            raise ValueError(f"No candidate found for park_id={park_id}, chosen_rank={chosen_rank}")

        row = subset.iloc[0]

        reviewed_rows.append({
            "park_id": int(row["park_id"]),
            "park_name": row.get("park_name", ""),
            "selected_name": row.get("osm_name", ""),
            "selected_score": row.get("score", None),
            "selected_contains_point": row.get("contains_point", None),
            "selected_candidate_rank": int(row.get("candidate_rank_local", chosen_rank)),
            "geometry": row.geometry,
        })

    gdf_reviewed = gpd.GeoDataFrame(reviewed_rows, geometry="geometry", crs=cand.crs)
    gdf_reviewed.to_file(OUT_MAIN8_REVIEWED, driver="GeoJSON")
    print(f"[OK] wrote {OUT_MAIN8_REVIEWED}")

    # Combine with original non-main8
    orig_keep = orig[~orig["park_id"].isin(MAIN8)].copy()
    full = pd.concat([orig_keep, gdf_reviewed], ignore_index=True)
    full = gpd.GeoDataFrame(full, geometry="geometry", crs=orig.crs)
    full.to_file(OUT_FULL_REVIEWED, driver="GeoJSON")
    print(f"[OK] wrote {OUT_FULL_REVIEWED}")

    print()
    print(gdf_reviewed.drop(columns="geometry").to_string(index=False))


if __name__ == "__main__":
    main()