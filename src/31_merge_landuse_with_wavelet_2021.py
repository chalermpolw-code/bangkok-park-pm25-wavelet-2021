# analysis/src/31_merge_landuse_with_wavelet_2021.py
# Merge Sentinel land-use (NDVI/NDBI) with Tier-A wavelet bandmeans (2021)
# Handles:
# - pairs file that has park_id + outside_station_id (no pair_id)
# - wavelet bandmeans that use column name "pair" (not pair_id)

import os
import pandas as pd

LANDUSE = "analysis/data_processed/landuse_2021/station_landuse_2021_buffers_NDVI_NDBI.csv"
PAIRS   = "analysis/outputs/pairs_2021_tierA.csv"
ATT_BAND = "analysis/outputs/wavelet_2021/attenuation_bandmeans.csv"
COH_BAND = "analysis/outputs/wavelet_2021/coherence_bandmeans.csv"

OUTDIR = "analysis/data_processed/landuse_2021"
os.makedirs(OUTDIR, exist_ok=True)
OUTFILE = os.path.join(OUTDIR, "pairs_landuse_wavelet_2021.csv")

MAIN_BUFFER_M = 500  # first analysis buffer
YEAR = 2021

BANDS = ["2-6h", "6-18h", "18-30h (diurnal)", "2-7d", "7-14d"]

def to_num_str(x) -> str:
    """Convert IDs like 09, 9, 9.0 safely -> '9'."""
    return str(int(float(str(x).strip())))

def ensure_wavelet_pair_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Wavelet bandmeans use 'pair' column -> rename to 'pair_id'."""
    df = df.copy()
    if "pair_id" in df.columns:
        return df
    if "pair" in df.columns:
        return df.rename(columns={"pair": "pair_id"})
    raise ValueError(f"[{name}] must contain 'pair' or 'pair_id'. Found: {list(df.columns)}")

def build_pair_id(park_id, outside_id) -> str:
    """Match your wavelet naming: park11_outside09_2021 (2-digit IDs)."""
    return f"park{int(park_id):02d}_outside{int(outside_id):02d}_{YEAR}"

def main():
    # --- Load landuse and pick buffer ---
    lu = pd.read_csv(LANDUSE)
    lu = lu[lu["buffer_m"] == MAIN_BUFFER_M].copy()

    # --- Load pairs and wavelet bandmeans ---
    pairs = pd.read_csv(PAIRS)
    att = ensure_wavelet_pair_id(pd.read_csv(ATT_BAND), "attenuation_bandmeans.csv")
    coh = ensure_wavelet_pair_id(pd.read_csv(COH_BAND), "coherence_bandmeans.csv")

    # --- Construct pair_id from pairs file ---
    required = {"park_id", "outside_station_id"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"[pairs_2021_tierA.csv] must contain {required}. Found: {list(pairs.columns)}")

    # Keep rank==1 only (safest). If rank column not present, keep all.
    if "rank" in pairs.columns:
        pairs = pairs[pairs["rank"] == 1].copy()

    pairs["pair_id"] = pairs.apply(lambda r: build_pair_id(r["park_id"], r["outside_station_id"]), axis=1)

    # numeric IDs for joining Sentinel landuse
    pairs["park_id_num"] = pairs["park_id"].apply(to_num_str)
    pairs["outside_id_num"] = pairs["outside_station_id"].apply(to_num_str)

    # --- Prepare Sentinel landuse tables ---
    lu_park = lu[lu["station_type"] == "park"].copy()
    lu_park["park_id_num"] = lu_park["station_id"].apply(to_num_str)
    lu_park = lu_park.rename(columns={
        "NDVI_mean": "park_NDVI",
        "NDBI_mean": "park_NDBI",
        "NDVI_count": "park_NDVI_n",
        "NDBI_count": "park_NDBI_n",
    })[["park_id_num", "park_NDVI", "park_NDBI", "park_NDVI_n", "park_NDBI_n"]]

    lu_out = lu[lu["station_type"] == "outside"].copy()
    lu_out["outside_id_num"] = lu_out["station_id"].apply(to_num_str)
    lu_out = lu_out.rename(columns={
        "NDVI_mean": "out_NDVI",
        "NDBI_mean": "out_NDBI",
        "NDVI_count": "out_NDVI_n",
        "NDBI_count": "out_NDBI_n",
    })[["outside_id_num", "out_NDVI", "out_NDBI", "out_NDVI_n", "out_NDBI_n"]]

    # --- Merge Sentinel landuse into pairs ---
    df = pairs.merge(lu_park, on="park_id_num", how="left").merge(lu_out, on="outside_id_num", how="left")

    # deltas (park - outside)
    df["dNDVI"] = df["park_NDVI"] - df["out_NDVI"]
    df["dNDBI"] = df["park_NDBI"] - df["out_NDBI"]

    # --- Keep only wavelet columns we need; rename to avoid collisions ---
    att_keep = ["pair_id"] + [c for c in BANDS if c in att.columns]
    coh_keep = ["pair_id"] + [c for c in BANDS if c in coh.columns]
    att = att[att_keep].rename(columns={b: f"att_{b}" for b in BANDS if b in att_keep})
    coh = coh[coh_keep].rename(columns={b: f"coh_{b}" for b in BANDS if b in coh_keep})

    # --- Merge wavelet outcomes ---
    df = df.merge(att, on="pair_id", how="left").merge(coh, on="pair_id", how="left")

    df["buffer_m"] = MAIN_BUFFER_M
    df.to_csv(OUTFILE, index=False)
    print(f"[OK] wrote {OUTFILE} (n={len(df)}, buffer_m={MAIN_BUFFER_M})")

if __name__ == "__main__":
    main()