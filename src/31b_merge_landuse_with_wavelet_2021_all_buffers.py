import os
import pandas as pd

LANDUSE = "analysis/data_processed/landuse_2021/station_landuse_2021_buffers_NDVI_NDBI.csv"
PAIRS   = "analysis/outputs/pairs_2021_tierA.csv"
ATT_BAND = "analysis/outputs/wavelet_2021/attenuation_bandmeans.csv"
COH_BAND = "analysis/outputs/wavelet_2021/coherence_bandmeans.csv"

OUTDIR = "analysis/data_processed/landuse_2021"
os.makedirs(OUTDIR, exist_ok=True)
OUTFILE = os.path.join(OUTDIR, "pairs_landuse_wavelet_2021_ALLBUFFERS.csv")

YEAR = 2021
BUFFERS = [250, 500, 1000]
BANDS = ["2-6h", "6-18h", "18-30h (diurnal)", "2-7d", "7-14d"]

def to_num_str(x) -> str:
    return str(int(float(str(x).strip())))

def ensure_pair_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pair_id" in df.columns:
        return df
    if "pair" in df.columns:
        return df.rename(columns={"pair": "pair_id"})
    raise ValueError(f"Wavelet file missing 'pair'/'pair_id': {list(df.columns)}")

def build_pair_id(park_id, outside_id) -> str:
    return f"park{int(park_id):02d}_outside{int(outside_id):02d}_{YEAR}"

def main():
    lu_all = pd.read_csv(LANDUSE)
    pairs = pd.read_csv(PAIRS)

    # keep nearest only
    if "rank" in pairs.columns:
        pairs = pairs[pairs["rank"] == 1].copy()

    pairs["pair_id"] = pairs.apply(lambda r: build_pair_id(r["park_id"], r["outside_station_id"]), axis=1)
    pairs["park_id_num"] = pairs["park_id"].apply(to_num_str)
    pairs["outside_id_num"] = pairs["outside_station_id"].apply(to_num_str)

    att = ensure_pair_id(pd.read_csv(ATT_BAND))
    coh = ensure_pair_id(pd.read_csv(COH_BAND))

    att = att[["pair_id"] + BANDS].rename(columns={b: f"att_{b}" for b in BANDS})
    coh = coh[["pair_id"] + BANDS].rename(columns={b: f"coh_{b}" for b in BANDS})

    rows = []
    for buf in BUFFERS:
        lu = lu_all[lu_all["buffer_m"] == buf].copy()

        lu_park = lu[lu["station_type"] == "park"].copy()
        lu_park["park_id_num"] = lu_park["station_id"].apply(to_num_str)
        lu_park = lu_park.rename(columns={
            "NDVI_mean": "park_NDVI", "NDBI_mean": "park_NDBI",
            "NDVI_count": "park_NDVI_n", "NDBI_count": "park_NDBI_n",
        })[["park_id_num", "park_NDVI", "park_NDBI", "park_NDVI_n", "park_NDBI_n"]]

        lu_out = lu[lu["station_type"] == "outside"].copy()
        lu_out["outside_id_num"] = lu_out["station_id"].apply(to_num_str)
        lu_out = lu_out.rename(columns={
            "NDVI_mean": "out_NDVI", "NDBI_mean": "out_NDBI",
            "NDVI_count": "out_NDVI_n", "NDBI_count": "out_NDBI_n",
        })[["outside_id_num", "out_NDVI", "out_NDBI", "out_NDVI_n", "out_NDBI_n"]]

        df = pairs.merge(lu_park, on="park_id_num", how="left").merge(lu_out, on="outside_id_num", how="left")
        df["dNDVI"] = df["park_NDVI"] - df["out_NDVI"]
        df["dNDBI"] = df["park_NDBI"] - df["out_NDBI"]
        df["buffer_m"] = buf

        df = df.merge(att, on="pair_id", how="left").merge(coh, on="pair_id", how="left")
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUTFILE, index=False)
    print(f"[OK] wrote {OUTFILE} (rows={len(out)})")

if __name__ == "__main__":
    main()