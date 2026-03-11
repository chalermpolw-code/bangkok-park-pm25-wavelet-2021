# analysis/src/30_export_sentinel_ndvi_ndbi_2021_gee.py
# Export Sentinel-2 NDVI + NDBI (2021) buffer means for ALL park + outside stations
# Output: analysis/data_processed/landuse_2021/station_landuse_2021_buffers_NDVI_NDBI.csv

import os
import pandas as pd
import ee

# -----------------------------
# CONFIG (edit only if needed)
# -----------------------------
PROJECT_ID = "ee-research-sau"

PARK_SITES_CSV = "analysis/data_raw/sites_parks_20_bkk_pm25.csv"
OUTSIDE_SITES_CSV = "analysis/data_raw/bkk_district_stations_50.csv"

# Your confirmed column names:
PARK_ID_COL = "park_id"
OUTSIDE_ID_COL = "station_id"
LAT_COL = "latitude"
LON_COL = "longitude"

DATE_START = "2021-01-01"
DATE_END = "2022-01-01"

BUFFERS_M = [250, 500, 1000]

OUTDIR = "analysis/data_processed/landuse_2021"
OUTFILE = os.path.join(OUTDIR, "station_landuse_2021_buffers_NDVI_NDBI.csv")

# -----------------------------
# Helpers
# -----------------------------
def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
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
    # Conservative SCL-based mask for Sentinel-2 SR
    scl = img.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return img.updateMask(good)

def add_indices(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return img.addBands([ndvi, ndbi])

def reduce_for_buffer(fc: ee.FeatureCollection, comp_img: ee.Image, buffer_m: int) -> ee.FeatureCollection:
    buff_fc = fc.map(lambda f: f.buffer(buffer_m).set({"buffer_m": buffer_m}))
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    out = comp_img.select(["NDVI", "NDBI"]).reduceRegions(
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
        .map(add_indices)
    )

    comp = s2.median()

    rows = []
    for b in BUFFERS_M:
        print(f"[INFO] Reducing buffers: {b} m ...")
        fc_red = reduce_for_buffer(fc_all, comp, b)

        # Small result (~70 stations per buffer) so we can bring it client-side
        feats = fc_red.getInfo()["features"]
        for f in feats:
            p = f["properties"]
            rows.append({
                "station_id": p.get("station_id"),
                "station_type": p.get("station_type"),
                "buffer_m": p.get("buffer_m"),
                "NDVI_mean": p.get("NDVI_mean"),
                "NDVI_count": p.get("NDVI_count"),
                "NDBI_mean": p.get("NDBI_mean"),
                "NDBI_count": p.get("NDBI_count"),
            })

    df = pd.DataFrame(rows).sort_values(["station_type", "station_id", "buffer_m"])
    df.to_csv(OUTFILE, index=False)
    print(f"[OK] wrote {OUTFILE} ({len(df)} rows)")

if __name__ == "__main__":
    main()