#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ingest Bangkok park XLSX (monthly sheets) into clean long-form hourly time series.

Input:
  analysis/data_raw/parks_xlsx_en/<YEAR_BE>/*.xlsx
  where YEAR_BE in {2564,2565,2566}, each file is one park, sheets are months like "01-2564".

Output:
  analysis/data_processed/parks_hourly/park01.csv ... park20.csv
  columns: park_id, t (Asia/Bangkok), pm25

Also writes:
  analysis/outputs/ingest_summary.csv
"""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IN_BASE = ROOT / "analysis" / "data_raw" / "parks_xlsx_en"
OUT_DIR = ROOT / "analysis" / "data_processed" / "parks_hourly"
SUM_DIR = ROOT / "analysis" / "outputs"

YEARS_BE = ["2564", "2565", "2566"]
TZ = "Asia/Bangkok"


TIME_RE = re.compile(r"^\d{2}:\d{2}$")  # 00:00 ... 23:00
SHEET_RE = re.compile(r"^(?P<mm>\d{2})-(?P<yyyy>\d{4})$")  # 01-2564


def be_to_ce(year_be: int) -> int:
    return year_be - 543


def park_id_from_filename(p: Path) -> str:
    # expects "01_....xlsx"
    m = re.match(r"^(?P<pid>\d{2})_", p.stem)
    if not m:
        raise ValueError(f"Cannot parse park_id from filename: {p.name}")
    return m.group("pid")


def parse_one_sheet(xlsx: Path, sheet_name: str) -> pd.DataFrame:
    """
    Reads the month sheet and returns long-form hourly data with columns:
      t (tz-aware), pm25
    """
    m = SHEET_RE.match(sheet_name.strip())
    if not m:
        return pd.DataFrame(columns=["t", "pm25"])

    mm = int(m.group("mm"))
    yy_be = int(m.group("yyyy"))
    yy = be_to_ce(yy_be)

    # Read entire sheet raw
    raw = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)

    # Expect row 1 contains day numbers (1..31) and col 0 is "เวลา/วันที่"
    # Row 0 is title text. Data starts row 2.
    if raw.shape[0] < 5 or raw.shape[1] < 5:
        return pd.DataFrame(columns=["t", "pm25"])

    day_row = raw.iloc[1, 1:]  # day numbers across columns 1..end
    days = pd.to_numeric(day_row, errors="coerce")
    day_cols = raw.columns[1:]

    # Data rows: keep only rows where first column looks like HH:MM
    times = raw.iloc[:, 0].astype(str).str.strip()
    mask_time = times.str.match(TIME_RE, na=False)
    data = raw.loc[mask_time, :].copy()

    if data.empty:
        return pd.DataFrame(columns=["t", "pm25"])

    # Build long table: columns are days, rows are hours
    data_times = data.iloc[:, 0].astype(str).str.strip()
    values = data.iloc[:, 1:]
    values.columns = day_cols

    # Melt to long: (time, day, value)
    long = values.copy()
    long.insert(0, "time_str", data_times.values)

    long = long.melt(
        id_vars=["time_str"],
        var_name="day_col",
        value_name="pm25"
    )

    # Map day_col -> day number
    day_map = dict(zip(day_cols, days.values))
    long["day"] = long["day_col"].map(day_map)

    # Drop invalid days and missing pm25
    long["day"] = pd.to_numeric(long["day"], errors="coerce")
    long["pm25"] = pd.to_numeric(long["pm25"], errors="coerce")
    long = long.dropna(subset=["day", "pm25"])

    # Build datetime
    # Combine yyyy-mm-day + HH:MM, localize to Bangkok
    long["day"] = long["day"].astype(int)
    dt_str = (
        long["day"].astype(str).str.zfill(2)
        .radd(f"{yy:04d}-{mm:02d}-")
        + " "
        + long["time_str"]
    )

    t = pd.to_datetime(dt_str, errors="coerce", format="%Y-%m-%d %H:%M")
    long["t"] = t.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    long = long.dropna(subset=["t"])

    return long[["t", "pm25"]].sort_values("t")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUM_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # Collect per-park across all years
    by_park = {}

    for y in YEARS_BE:
        folder = IN_BASE / y
        files = sorted(folder.glob("*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No XLSX files found in {folder}")

        for f in files:
            pid = park_id_from_filename(f)
            try:
                xls = pd.ExcelFile(f)
                sheets = xls.sheet_names
            except Exception as e:
                raise RuntimeError(f"Failed to open {f}: {e}") from e

            parts = []
            for sh in sheets:
                part = parse_one_sheet(f, sh)
                if not part.empty:
                    parts.append(part)

            if parts:
                df = pd.concat(parts, ignore_index=True)
                df = df.drop_duplicates(subset=["t"]).sort_values("t")
            else:
                df = pd.DataFrame(columns=["t", "pm25"])

            df.insert(0, "park_id", pid)

            by_park.setdefault(pid, []).append(df)

            summary_rows.append({
                "year_be": y,
                "park_id": pid,
                "file": f.name,
                "n_rows": len(df),
                "t_min": df["t"].min() if len(df) else None,
                "t_max": df["t"].max() if len(df) else None,
            })

    # Write per-park combined CSV
    for pid, chunks in sorted(by_park.items()):
        dfp = pd.concat(chunks, ignore_index=True)
        dfp = dfp.drop_duplicates(subset=["t"]).sort_values("t")
        out = OUT_DIR / f"park{pid}.csv"
        dfp.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote {out} ({len(dfp):,} rows)")

    # Write summary
    summ = pd.DataFrame(summary_rows).sort_values(["park_id", "year_be"])
    summ_out = SUM_DIR / "ingest_summary.csv"
    summ.to_csv(summ_out, index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {summ_out}")


if __name__ == "__main__":
    main()
