#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "analysis" / "data_raw" / "outside_pm25_xlsx"
OUTDIR = ROOT / "analysis" / "data_processed" / "outside_hourly"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY = ROOT / "analysis" / "outputs" / "outside_ingest_summary.csv"
SUMMARY.parent.mkdir(parents=True, exist_ok=True)

YEARS = ["2564", "2565", "2566"]
TZ = "Asia/Bangkok"

# Skip clearly non-PM sheets (noise etc.)
SKIP_PATTERNS = ["sound", "noise", "เสียง", "leq", "lmax"]

PM_KEYS = ["pm2.5", "pm25", "pm2_5", "pm2·5", "pm2,5"]

def station_id_from_name(name: str) -> str:
    m = re.match(r"^\s*(\d{1,2})\s*[_\-]", name)
    return m.group(1).zfill(2) if m else None

def norm(s):
    return str(s).strip().lower().replace(" ", "")

def should_skip_sheet(sheet_name: str) -> bool:
    s = sheet_name.lower()
    return any(p in s for p in SKIP_PATTERNS)

def find_pm25_col(cols):
    for c in cols:
        cc = norm(c)
        if any(k in cc for k in PM_KEYS):
            return c
    return None

def parse_datetime(df):
    cols = list(df.columns)

    # Case 1: Thai split date/time
    c_date = None
    c_time = None
    for c in cols:
        cc = norm(c)
        if "วันที่" in cc or cc == "date":
            c_date = c
        if "เวลา" in cc or cc == "time":
            c_time = c

    if c_date is not None and c_time is not None:
        dt = pd.to_datetime(
            df[c_date].astype(str).str.strip() + " " + df[c_time].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True
        )
        if dt.notna().mean() > 0.6:
            return dt

    # Case 2: single datetime column named Time/Datetime/Date
    for c in cols:
        cc = norm(c)
        if cc in ["time", "datetime", "timestamp", "date"]:
            dt = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if dt.notna().mean() > 0.6:
                return dt

    # Case 3: first column is datetime
    dt = pd.to_datetime(df.iloc[:, 0], errors="coerce", dayfirst=True)
    if dt.notna().mean() > 0.6:
        return dt

    return None

def read_sheet_best(xlsx, sheet):
    # Try header rows 0..6, pick best one by “datetime parseability + pm25 col”
    best = None
    best_score = -1

    for h in [0,1,2,3,4,5,6]:
        try:
            df = pd.read_excel(xlsx, sheet_name=sheet, header=h)
            pm = find_pm25_col(df.columns)
            dt = parse_datetime(df)
            score = 0
            if pm is not None: score += 2
            if dt is not None: score += 2
            # penalize unnamed headers
            unnamed = sum("unnamed" in norm(c) for c in df.columns)
            score -= unnamed * 0.2

            if score > best_score:
                best_score = score
                best = (h, df)
        except Exception:
            continue

    return best  # (header_row, df) or None

def main():
    acc = {}  # sid -> list of chunks
    summary_rows = []

    for y in YEARS:
        folder = BASE / y
        if not folder.exists():
            continue

        for xlsx in sorted(folder.glob("*.xls*")):
            sid = station_id_from_name(xlsx.name)
            if sid is None:
                continue

            try:
                xl = pd.ExcelFile(xlsx)
                sheets = xl.sheet_names
            except Exception as e:
                summary_rows.append([y, sid, xlsx.name, "ERROR_OPEN", "", "", str(e)])
                continue

            n_used = 0
            n_skipped = 0
            n_failed = 0

            for sh in sheets:
                if should_skip_sheet(sh):
                    n_skipped += 1
                    continue

                best = read_sheet_best(xlsx, sh)
                if best is None:
                    n_failed += 1
                    continue

                h, df = best
                pmcol = find_pm25_col(df.columns)
                dt = parse_datetime(df)
                if pmcol is None or dt is None:
                    n_failed += 1
                    continue

                out = pd.DataFrame({
                    "time": dt,
                    "pm25": pd.to_numeric(df[pmcol], errors="coerce")
                }).dropna(subset=["time"])

                # localize/convert timezone
                out["time"] = pd.to_datetime(out["time"], errors="coerce", dayfirst=True)
                out = out.dropna(subset=["time"])
                if out["time"].dt.tz is None:
                    out["time"] = out["time"].dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
                else:
                    out["time"] = out["time"].dt.tz_convert(TZ)

                out["outside_station_id"] = sid
                out = out.drop_duplicates(subset=["time"], keep="last")
                out = out.sort_values("time")

                if len(out) > 0:
                    acc.setdefault(sid, []).append(out)
                    n_used += 1
                else:
                    n_failed += 1

            summary_rows.append([y, sid, xlsx.name, "OK", len(sheets), n_used, f"skipped={n_skipped}, failed={n_failed}"])

    # Write per-station CSV
    for sid, parts in sorted(acc.items()):
        df = pd.concat(parts, ignore_index=True)
        df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
        out = OUTDIR / f"outside_{sid}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[OK] wrote {out} ({len(df):,} rows)")

    # Write summary
    pd.DataFrame(summary_rows, columns=[
        "year","outside_station_id","file","status","n_sheets","n_used_sheets","notes_or_error"
    ]).to_csv(SUMMARY, index=False, encoding="utf-8-sig")

    print(f"[DONE] wrote {SUMMARY}")
    print("[DONE] outside ingestion complete.")

if __name__ == "__main__":
    main()
