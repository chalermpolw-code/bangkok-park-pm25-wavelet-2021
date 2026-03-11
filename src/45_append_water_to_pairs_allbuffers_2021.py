#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Append station-level water metrics to existing pair-level ALLBUFFERS file.

Input
-----
1) analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv
2) analysis/data_processed/landuse_2021/station_water_2021_buffers_NDWI_MNDWI_WATER.csv

Output
------
analysis/data_processed/landuse_2021/pairs_landuse_water_wavelet_2021_ALLBUFFERS.csv
"""

import os
import pandas as pd

IN_PAIRS = "analysis/data_processed/landuse_2021/pairs_landuse_wavelet_2021_ALLBUFFERS.csv"
IN_WATER = "analysis/data_processed/landuse_2021/station_water_2021_buffers_NDWI_MNDWI_WATER.csv"
OUT_CSV = "analysis/data_processed/landuse_2021/pairs_landuse_water_wavelet_2021_ALLBUFFERS.csv"


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    pairs = pd.read_csv(IN_PAIRS)
    water = pd.read_csv(IN_WATER)

    # Normalize keys
    pairs["park_id_key"] = pairs["park_id"].astype(str)
    pairs["outside_station_id_key"] = pairs["outside_station_id"].astype(str)
    pairs["buffer_m"] = pairs["buffer_m"].astype(int)

    water["station_id_key"] = water["station_id"].astype(str)
    water["buffer_m"] = water["buffer_m"].astype(int)

    # Split park/outside station metrics
    park_w = water[water["station_type"] == "park"].copy()
    out_w = water[water["station_type"] == "outside"].copy()

    park_w = park_w.rename(columns={
        "station_id_key": "park_id_key",
        "NDWI_mean": "park_NDWI",
        "NDWI_count": "park_NDWI_n",
        "MNDWI_mean": "park_MNDWI",
        "MNDWI_count": "park_MNDWI_n",
        "WATER_FRAC_mean": "park_WATER_FRAC",
        "WATER_FRAC_count": "park_WATER_FRAC_n",
    })

    out_w = out_w.rename(columns={
        "station_id_key": "outside_station_id_key",
        "NDWI_mean": "out_NDWI",
        "NDWI_count": "out_NDWI_n",
        "MNDWI_mean": "out_MNDWI",
        "MNDWI_count": "out_MNDWI_n",
        "WATER_FRAC_mean": "out_WATER_FRAC",
        "WATER_FRAC_count": "out_WATER_FRAC_n",
    })

    keep_park = [
        "park_id_key", "buffer_m",
        "park_NDWI", "park_NDWI_n",
        "park_MNDWI", "park_MNDWI_n",
        "park_WATER_FRAC", "park_WATER_FRAC_n",
    ]
    keep_out = [
        "outside_station_id_key", "buffer_m",
        "out_NDWI", "out_NDWI_n",
        "out_MNDWI", "out_MNDWI_n",
        "out_WATER_FRAC", "out_WATER_FRAC_n",
    ]

    out = pairs.merge(park_w[keep_park], on=["park_id_key", "buffer_m"], how="left")
    out = out.merge(out_w[keep_out], on=["outside_station_id_key", "buffer_m"], how="left")

    out = out.drop(columns=["park_id_key", "outside_station_id_key"])
    out.to_csv(OUT_CSV, index=False)

    print(f"[DONE] wrote {OUT_CSV}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()