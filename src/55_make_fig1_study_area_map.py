from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.lines import Line2D
from adjustText import adjust_text

# ============================================================
# PATH CONFIG
# ============================================================

PROJECT_ROOT = Path("/Users/chalermpolw/Desktop/raw_data/urban_rhythm_damping_green_parks_bkk_pm25_wavelet_regimes 2")

DATA_RAW = PROJECT_ROOT / "analysis" / "data_raw"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARK_SITES_CSV = DATA_RAW / "sites_parks_20_bkk_pm25.csv"
OUTSIDE_SITES_CSV = DATA_RAW / "bkk_district_stations_50.csv"
PAIRS_TIERA_CSV = PROJECT_ROOT / "analysis" / "outputs" / "pairs_2021_tierA.csv"

ADMIN2_SHP = DATA_RAW / "bangkok_admin2" / "tha_admin2.shp"

OUT_PNG = OUTPUT_DIR / "Fig1_study_area_map.png"
OUT_PDF = OUTPUT_DIR / "Fig1_study_area_map.pdf"

# ============================================================
# SETTINGS
# ============================================================

BANGKOK_PCODE = "TH10"
FIGSIZE = (10.5, 10.5)
DPI = 300

# Refined color palette for better contrast
COLOR_DISTRICT_FILL = "#f8f9fa"
COLOR_DISTRICT_EDGE = "#dee2e6"
COLOR_OUTSIDE_ALL = "#adb5bd"      # Muted grey for unused outside
COLOR_PARK_ALL = "#74c476"         # Soft green for unused parks
COLOR_RETAINED_OUTSIDE = "#e6550d" # Bold orange for selected outside
COLOR_RETAINED_PARK = "#006d2c"    # Dark forest green for selected parks

# ============================================================
# THAI -> ENGLISH NAME MAPS
# ============================================================

PARK_NAME_MAP = {
    "สวนจตุจักร": "Chatuchak Park",
    "สวนพระนคร": "Phra Nakhon Park",
    "สวนลุมพินี": "Lumpini Park",
    "สวนวชิรเบญจทัศ": "Wachirabenchathat Park",
    "สวนสมเด็จพระนางเจ้าสิริกิติ์ฯ": "Queen Sirikit Park",
    "สวนหนองจอก": "Nong Chok Park",
    "สวนหลวงร.9": "Suan Luang Rama IX Park",
    "สวนหลวง ร.๙": "Suan Luang Rama IX Park",
    "สวนหลวง ร.9": "Suan Luang Rama IX Park",
    "สวนเสรีไทย": "Seri Thai Park",
}

DISTRICT_NAME_MAP = {
    "เขตปทุมวัน": "Pathum Wan District",
    "เขตบางซื่อ": "Bang Sue District",
    "เขตลาดกระบัง": "Lat Krabang District",
    "เขตหนองจอก": "Nong Chok District",
    "เขตบึงกุ่ม": "Bueng Kum District",
    "เขตประเวศ": "Prawet District",
    "เขตพระนคร": "Phra Nakhon District",
    "เขตบางเขน": "Bang Khen District",
    "เขตบางคอแหลม": "Bang Kho Laem District",
    "เขตดอนเมือง": "Don Mueang District",
    "เขตสะพานสูง": "Saphan Sung District",
    "เขตบางกอกน้อย": "Bangkok Noi District",
    "เขตทวีวัฒนา": "Thawi Watthana District",
    "เขตบางแค": "Bang Khae District",
    "เขตทุ่งครุ": "Thung Khru District",
    "เขตราชเทวี": "Ratchathewi District",
    " เขตวัฒนา": "Watthana District",
    "เขตวัฒนา": "Watthana District",
}

# ============================================================
# HELPERS
# ============================================================

def normalize_id(value) -> str:
    s = str(value).strip()
    if s.lower().startswith("park"):
        digits = re.findall(r"\d+", s)
        return digits[0].zfill(2) if digits else s
    digits = re.findall(r"\d+", s)
    return digits[0].zfill(2) if digits else s

def translate_park_name(th_name: str) -> str:
    th_name = str(th_name).strip()
    return PARK_NAME_MAP.get(th_name, th_name)

def translate_district_name(th_name: str) -> str:
    th_name = str(th_name).strip()
    return DISTRICT_NAME_MAP.get(th_name, th_name)

def load_sites() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    parks = pd.read_csv(PARK_SITES_CSV)
    outside = pd.read_csv(OUTSIDE_SITES_CSV)

    parks["park_id_std"] = parks["park_id"].apply(normalize_id)
    parks["park_name_std"] = parks["park_name_th"].apply(translate_park_name)
    parks["district_name_std"] = parks["district_th"].apply(translate_district_name)

    gparks = gpd.GeoDataFrame(
        parks.copy(),
        geometry=gpd.points_from_xy(parks["longitude"], parks["latitude"]),
        crs="EPSG:4326",
    )

    outside["outside_id_std"] = outside["station_id"].apply(normalize_id)
    outside["outside_name_std"] = outside["district_th"].apply(translate_district_name)

    goutside = gpd.GeoDataFrame(
        outside.copy(),
        geometry=gpd.points_from_xy(outside["longitude"], outside["latitude"]),
        crs="EPSG:4326",
    )

    return gparks, goutside

def load_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(PAIRS_TIERA_CSV)
    pairs["park_id_std"] = pairs["park_id"].apply(normalize_id)
    pairs["outside_id_std"] = pairs["outside_station_id"].apply(normalize_id)
    return pairs

def load_bangkok_districts() -> gpd.GeoDataFrame:
    adm2 = gpd.read_file(ADMIN2_SHP)
    adm2.columns = [c.lower() for c in adm2.columns]

    if "adm1_pcode" in adm2.columns:
        bkk = adm2[adm2["adm1_pcode"] == BANGKOK_PCODE].copy()
    elif "adm1_en" in adm2.columns:
        bkk = adm2[adm2["adm1_en"].str.lower() == "bangkok"].copy()
    else:
        raise KeyError("Could not find adm1_pcode or adm1_en in ADM2 shapefile.")

    return bkk


def main():
    gparks, goutside = load_sites()
    pairs = load_pairs()
    bkk_districts = load_bangkok_districts()

    retained_park_ids = sorted(pairs["park_id_std"].unique().tolist())
    retained_outside_ids = sorted(pairs["outside_id_std"].unique().tolist())

    retained_parks = gparks[gparks["park_id_std"].isin(retained_park_ids)].copy()
    retained_outside = goutside[goutside["outside_id_std"].isin(retained_outside_ids)].copy()

    target_crs = "EPSG:32647"
    bkk_districts_p = bkk_districts.to_crs(target_crs)
    gparks_p = gparks.to_crs(target_crs)
    goutside_p = goutside.to_crs(target_crs)
    retained_parks_p = retained_parks.to_crs(target_crs)
    retained_outside_p = retained_outside.to_crs(target_crs)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bkk_districts_p.plot(
        ax=ax,
        color=COLOR_DISTRICT_FILL,
        edgecolor=COLOR_DISTRICT_EDGE,
        linewidth=0.8,
        zorder=1,
    )

    goutside_p.plot(
        ax=ax,
        color=COLOR_OUTSIDE_ALL,
        markersize=20,
        alpha=0.60,
        zorder=2,
    )

    gparks_p.plot(
        ax=ax,
        color=COLOR_PARK_ALL,
        markersize=20,
        alpha=0.60,
        zorder=3,
    )

    retained_outside_p.plot(
        ax=ax,
        color=COLOR_RETAINED_OUTSIDE,
        markersize=45,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

    retained_parks_p.plot(
        ax=ax,
        color=COLOR_RETAINED_PARK,
        markersize=45,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

    halo = [pe.withStroke(linewidth=2.5, foreground="white")]

    # List to hold all text objects for adjustText
    texts = []

    # Create text objects for retained parks
    for _, row in retained_parks_p.iterrows():
        name = str(row["park_name_std"])
        x, y = row.geometry.x, row.geometry.y
        texts.append(
            ax.text(x, y, name, fontsize=8.5, color=COLOR_RETAINED_PARK,
                    fontweight="bold", zorder=6, path_effects=halo)
        )

    # Create text objects for retained outside districts
    for _, row in retained_outside_p.iterrows():
        name = str(row["outside_name_std"])
        x, y = row.geometry.x, row.geometry.y
        texts.append(
            ax.text(x, y, name, fontsize=8.5, color=COLOR_RETAINED_OUTSIDE,
                    fontweight="bold", zorder=6, path_effects=halo)
        )

    # Magically repel overlapping text and draw connecting lines
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.8, alpha=0.7, zorder=1)
    )

    xmin, ymin, xmax, ymax = bkk_districts_p.total_bounds
    xpad = (xmax - xmin) * 0.03
    ypad = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_axis_off()

    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_PARK_ALL, markeredgecolor='none',
               markersize=7, label='All 20 park-interior stations'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_OUTSIDE_ALL, markeredgecolor='none',
               markersize=7, label='All 50 outside stations'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_RETAINED_PARK, markeredgecolor='white',
               markersize=9, label='Retained Tier-A park stations'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_RETAINED_OUTSIDE, markeredgecolor='white',
               markersize=9, label='Retained Tier-A outside stations'),
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=9,
        title="Map layers",
        title_fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {OUT_PNG}")
    print(f"[OK] wrote {OUT_PDF}")


if __name__ == "__main__":
    main()