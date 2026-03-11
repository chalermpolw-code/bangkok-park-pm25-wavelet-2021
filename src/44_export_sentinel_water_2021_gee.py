#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Sentinel-2 water-related metrics (2021) for ALL park + outside stations.

Metrics
-------
- NDWI   = (B3 - B8)  / (B3 + B8)
- MNDWI  = (B3 - B11) / (B3 + B11)
- WATER_FRAC = fraction of pixels classified as likely water

Water rule (configurable)
-------------------------
A pixel is counted as water if:
- MNDWI > 0
- NDVI  < 0.20
- NDBI  < 0

This is intentionally conservative to reduce false positives from built/shadow.

Output
------
analysis/data_processed/landuse_2021/station_water_2021_buffers_NDWI_MNDWI_WATER.csv
"""

import os
import pandas as pd
import ee

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = "ee-research-sau"

PARK_SITES_CSV = "analysis/data_raw/sites_parks_20_bkk_pm25.csv"
OUTSIDE_SITES_CSV = "analysis/data_raw/bkk_district_stations_50.csv"

PARK_ID_COL = "park_id"
OUTSIDE_ID_COL = "station_id"
LAT_COL = "latitude"
LON_COL = "longitude"

DATE_START = "2021-01-01"
DATE_END = "2022-01-01"

BUFFERS_M = [250, 500, 1000]

OUTDIR = "analysis/data_processed/landuse_2021"
OUTFILE = os.path.join(OUTDIR, "station_water_2021_buffers_NDWI_MNDWI_WATER.csv")


# -----------------------------
# Helpers
# -----------------------------
def require_columns(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}. Found: {list(df.columns)}")


def to_feature_collection(df: pd.DataFrame, id_col: str, lat_col: str, lon_col: str, station_type: str) -> ee.FeatureCollection:
    feats = []
    for _, r in df.iterrows():
        sid = str(r[id_col])
        lat = float(r[lat_col])
        lon = float(r[lon_col])
        geom = ee.Geometry.Point([lon, lat])
        feats.append(ee.Feature(geom, {"station_id": sid, "station_type": station_type}))
    return ee.FeatureCollection(feats)


def mask_s2_sr(img: ee.Image) -> ee.Image:
    # Conservative SCL-based mask
    scl = img.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return img.updateMask(good)


def add_water_metrics(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = img.normalizedDifference(["B3", "B11"]).rename("MNDWI")

    water_mask = (
        mndwi.gt(0)
        .And(ndvi.lt(0.20))
        .And(ndbi.lt(0))
        .rename("WATER_FRAC")
        .toFloat()
    )

    return img.addBands([ndvi, ndbi, ndwi, mndwi, water_mask])


def reduce_for_buffer(fc: ee.FeatureCollection, comp_img: ee.Image, buffer_m: int) -> ee.FeatureCollection:
    buff_fc = fc.map(lambda f: f.buffer(buffer_m).set({"buffer_m": buffer_m}))
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    out = comp_img.select(["NDWI", "MNDWI", "WATER_FRAC"]).reduceRegions(
        collection=buff_fc,
        reducer=reducer,
        scale=10,
        tileScale=4
    )
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("[INFO] Initializing Earth Engine...")
    ee.Initialize(project=PROJECT_ID)
    print(f"[OK] EE initialized with project='{PROJECT_ID}'")

    print("[INFO] Reading station CSVs...")
    parks = pd.read_csv(PARK_SITES_CSV)
    outside = pd.read_csv(OUTSIDE_SITES_CSV)

    require_columns(parks, [PARK_ID_COL, LAT_COL, LON_COL], "parks")
    require_columns(outside, [OUTSIDE_ID_COL, LAT_COL, LON_COL], "outside")

    print(f"[INFO] Parks: {len(parks)} rows | Outside: {len(outside)} rows")

    fc_parks = to_feature_collection(parks, PARK_ID_COL, LAT_COL, LON_COL, "park")
    fc_out = to_feature_collection(outside, OUTSIDE_ID_COL, LAT_COL, LON_COL, "outside")
    fc_all = fc_parks.merge(fc_out)

    print("[INFO] Building Sentinel-2 SR (HARMONIZED) 2021 composite (median, cloud-masked)...")
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(mask_s2_sr)
        .map(add_water_metrics)
    )

    comp = s2.median()

    rows = []
    for b in BUFFERS_M:
        print(f"[INFO] Reducing buffers: {b} m ...")
        fc_red = reduce_for_buffer(fc_all, comp, b)

        feats = fc_red.getInfo()["features"]
        for f in feats:
            p = f["properties"]
            rows.append({
                "station_id": p.get("station_id"),
                "station_type": p.get("station_type"),
                "buffer_m": p.get("buffer_m"),
                "NDWI_mean": p.get("NDWI_mean"),
                "NDWI_count": p.get("NDWI_count"),
                "MNDWI_mean": p.get("MNDWI_mean"),
                "MNDWI_count": p.get("MNDWI_count"),
                "WATER_FRAC_mean": p.get("WATER_FRAC_mean"),
                "WATER_FRAC_count": p.get("WATER_FRAC_count"),
            })

    df = pd.DataFrame(rows).sort_values(["station_type", "station_id", "buffer_m"])
    df.to_csv(OUTFILE, index=False)
    print(f"[OK] wrote {OUTFILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()