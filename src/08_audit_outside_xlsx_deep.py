#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "analysis" / "data_raw" / "outside_pm25_xlsx"
OUT  = ROOT / "analysis" / "outputs" / "outside_xlsx_audit_deep.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

YEARS = ["2564", "2565", "2566"]

def station_id_from_name(name: str) -> str:
    # expects prefix like "09_" or "9_" at beginning
    m = re.match(r"^\s*(\d{1,2})\s*[_\-]", name)
    return m.group(1).zfill(2) if m else "NA"

def read_peek(xlsx: Path, sheet: str, header_try: int):
    # small peek to detect header row / columns
    df = pd.read_excel(xlsx, sheet_name=sheet, header=header_try, nrows=10)
    return [str(c).strip() for c in df.columns]

def main():
    rows = []
    for y in YEARS:
        folder = BASE / y
        if not folder.exists():
            continue

        for xlsx in sorted(folder.glob("*.xls*")):
            sid = station_id_from_name(xlsx.name)
            try:
                xl = pd.ExcelFile(xlsx)
                sheets = xl.sheet_names
            except Exception as e:
                rows.append([y, sid, xlsx.name, "ERROR_OPEN", "", "", "", str(e)])
                continue

            # check up to first 6 sheets (enough to characterize structure)
            for sh in sheets[:6]:
                # try a few header rows (0..6) because many files have title rows
                best_cols = None
                best_h = None
                err = ""
                for h in [0, 1, 2, 3, 4, 5, 6]:
                    try:
                        cols = read_peek(xlsx, sh, h)
                        # heuristic: a "real" header has multiple non-"Unnamed" cols
                        real = sum([(c and "unnamed" not in c.lower()) for c in cols])
                        if best_cols is None or real > sum([(c and "unnamed" not in c.lower()) for c in best_cols]):
                            best_cols = cols
                            best_h = h
                    except Exception as e:
                        err = str(e)

                rows.append([
                    y, sid, xlsx.name,
                    "OK" if best_cols is not None else "ERROR_READ",
                    len(sheets),
                    sh,
                    best_h if best_h is not None else "",
                    "|".join(best_cols[:20]) if best_cols is not None else err
                ])

    audit = pd.DataFrame(rows, columns=[
        "year","outside_station_id","file","status",
        "n_sheets","sample_sheet","best_header_row_guess","sample_cols_or_error"
    ])
    audit.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {OUT}")
    print(audit.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
