#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build mapping: each park station -> nearest outside (district) station by distance.

Inputs:
  analysis/data_raw/sites_parks_20_bkk_pm25.csv
  analysis/data_raw/bkk_district_stations_50.csv

Outputs:
  analysis/data_processed/park_to_outside_station.csv
  analysis/data_processed/park_to_outside_station_top3.csv
"""

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PARKS_CSV = ROOT / "analysis" / "data_raw" / "sites_parks_20_bkk_pm25.csv"
OUTSIDE_CSV = ROOT / "analysis" / "data_raw" / "bkk_district_stations_50.csv"

OUT_DIR = ROOT / "analysis" / "data_processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT1 = OUT_DIR / "park_to_outside_station.csv"
OUT3 = OUT_DIR / "park_to_outside_station_top3.csv"

def haversine_km(lat1, lon1, lat2, lon2):
    # Earth radius (km)
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def main():
    parks = pd.read_csv(PARKS_CSV)
    outs = pd.read_csv(OUTSIDE_CSV)

    # normalize column names (parks file can vary)
    parks_cols = {c.lower(): c for c in parks.columns}
    outs_cols = {c.lower(): c for c in outs.columns}

    def col(df_map, *names):
        for n in names:
            if n in df_map:
                return df_map[n]
        raise KeyError(f"Missing expected column. Tried: {names}")

    p_lat = col(parks_cols, "latitude", "lat")
    p_lon = col(parks_cols, "longitude", "lon", "lng")
    # park identifier
    p_id = parks_cols.get("park_id", None) or parks_cols.get("id", None) or parks.columns[0]

    o_lat = col(outs_cols, "latitude", "lat")
    o_lon = col(outs_cols, "longitude", "lon", "lng")
    o_id  = outs_cols.get("station_id", None) or outs_cols.get("id", None) or outs.columns[0]

    # optional descriptive columns
    o_dist = outs_cols.get("district_th", None)
    o_site = outs_cols.get("site_th", None)
    o_type = outs_cols.get("station_type_th", None)

    rows_best = []
    rows_top3 = []

    for _, pr in parks.iterrows():
        plat = float(pr[p_lat])
        plon = float(pr[p_lon])
        pid = pr[p_id]

        dists = []
        for _, orow in outs.iterrows():
            olat = float(orow[o_lat])
            olon = float(orow[o_lon])
            dist_km = haversine_km(plat, plon, olat, olon)
            d = {
                "park_id": pid,
                "park_lat": plat,
                "park_lon": plon,
                "outside_station_id": orow[o_id],
                "outside_lat": olat,
                "outside_lon": olon,
                "distance_km": dist_km,
            }
            if o_dist: d["outside_district_th"] = orow[o_dist]
            if o_site: d["outside_site_th"] = orow[o_site]
            if o_type: d["outside_type_th"] = orow[o_type]
            dists.append(d)

        ddf = pd.DataFrame(dists).sort_values("distance_km")

        # best
        rows_best.append(ddf.iloc[0].to_dict())

        # top3
        top3 = ddf.head(3).copy()
        top3["rank"] = [1, 2, 3]
        rows_top3.append(top3)

    best_df = pd.DataFrame(rows_best)
    top3_df = pd.concat(rows_top3, ignore_index=True)

    best_df.to_csv(OUT1, index=False, encoding="utf-8-sig")
    top3_df.to_csv(OUT3, index=False, encoding="utf-8-sig")

    print(f"[DONE] wrote {OUT1}")
    print(f"[DONE] wrote {OUT3}")
    print("\nPreview (first 5):")
    print(best_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
