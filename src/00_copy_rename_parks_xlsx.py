#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copy + rename Bangkok park XLSX files into an English-named mirror folder.

Assumption (based on your setup):
- You have already reordered sites_parks_20_bkk_pm25.csv so that ROW ORDER
  matches the intended park file order (1..20).
- Thai raw files are in: analysis/data_raw/parks_xlsx/<YEAR>/*.xlsx
- Output will be in:     analysis/data_raw/parks_xlsx_en/<YEAR>/*.xlsx

What it does:
- Reads your CSV in its current row order (1..20)
- Builds a safe English-ish slug from park_name_th + district_th
- Lists XLSX files in each YEAR folder and sorts by leading number in filename (e.g., "1.", "2.", ...)
- Copies each file i -> <YEAR>/XX_<slug>.xlsx
- Writes a mapping file: analysis/data_raw/parks_file_mapping.csv
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]  # project root (…/urban_rhythm_…/)
CSV_PATH = ROOT / "analysis" / "data_raw" / "sites_parks_20_bkk_pm25.csv"

SRC_BASE = ROOT / "analysis" / "data_raw" / "parks_xlsx"
DST_BASE = ROOT / "analysis" / "data_raw" / "parks_xlsx_en"

YEARS = ["2564", "2565", "2566"]


def leading_int(name: str) -> int:
    """
    Extract leading integer from filenames like:
    '1.สวน....xlsx' or '1.สวน....xlsx' or '01_....xlsx'
    If none found, return large number to push to end.
    """
    m = re.match(r"^\s*(\d+)", name)
    return int(m.group(1)) if m else 10**9


def slugify_th_to_ascii(s: str) -> str:
    """
    Make a filesystem-safe slug.
    We keep it simple: lower, replace spaces with underscore,
    remove symbols, and keep ASCII where possible.
    Thai will be removed by the ASCII filter, so we also keep a short fallback.
    """
    s = str(s).strip().lower()
    s = s.replace("ฯ", "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)  # keep ASCII only
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_slug(row: pd.Series, i: int) -> str:
    # Try to use Thai fields -> ASCII-safe slug; if it becomes empty, fall back to parkXX
    park_th = row.get("park_name_th", "")
    dist_th = row.get("district_th", "")
    base = f"{park_th}_{dist_th}"
    slug = slugify_th_to_ascii(base)
    if not slug:
        slug = f"park{i:02d}"
    return slug


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if len(df) != 20:
        raise ValueError(f"Expected 20 rows in CSV, got {len(df)}. Please check {CSV_PATH.name}")

    # Precompute slugs by row order
    slugs = [build_slug(df.iloc[i], i + 1) for i in range(20)]

    mapping_rows = []

    for year in YEARS:
        src_dir = SRC_BASE / year
        dst_dir = DST_BASE / year
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob("*.xlsx"), key=lambda p: leading_int(p.name))

        if len(files) != 20:
            raise ValueError(f"[{year}] Expected 20 XLSX files in {src_dir}, got {len(files)}")

        for i, src in enumerate(files, start=1):
            slug = slugs[i - 1]
            dst_name = f"{i:02d}_{slug}.xlsx"
            dst = dst_dir / dst_name

            shutil.copy2(src, dst)

            row = df.iloc[i - 1]
            mapping_rows.append(
                {
                    "year": year,
                    "index_01_20": i,
                    "src_filename": src.name,
                    "dst_filename": dst_name,
                    "district_th": row.get("district_th", ""),
                    "park_name_th": row.get("park_name_th", ""),
                    "latitude": row.get("latitude", ""),
                    "longitude": row.get("longitude", ""),
                }
            )

        print(f"[OK] {year}: copied+renamed 20 files -> {dst_dir}")

    out_map = ROOT / "analysis" / "data_raw" / "parks_file_mapping.csv"
    pd.DataFrame(mapping_rows).to_csv(out_map, index=False, encoding="utf-8-sig")
    print(f"[DONE] Wrote mapping: {out_map}")


if __name__ == "__main__":
    main()
