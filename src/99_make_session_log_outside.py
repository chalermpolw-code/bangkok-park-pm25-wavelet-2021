#!/usr/bin/env python3
"""
Make a clean, reproducible Markdown session log for the "outside-park" pipeline.

What it does
- Reads key summary CSVs (if present).
- Writes a single Markdown log file with:
  - Steps table (scripts/outputs)
  - Status counts (from outside_hourly_clean_summary.csv)
  - Tier A/B/C/D lists (heuristic based on NA fraction + step counts + status)
  - Quick index of important outputs

Run
  python analysis/src/99_make_session_log_outside.py

Output
  analysis/outputs/session_log_YYYY-MM-DD.md
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import pandas as pd


# -----------------------------
# Config (edit if you change paths)
# -----------------------------
ROOT = Path(".")  # run from repo root
OUTDIR = ROOT / "analysis" / "outputs"

CSV_OUTSIDE_AUDIT = OUTDIR / "outside_xlsx_audit_deep.csv"
CSV_OUTSIDE_INGEST_SUMMARY = OUTDIR / "outside_ingest_summary.csv"
CSV_QC_OUTSIDE = OUTDIR / "qc_outside_hourly_summary.csv"
CSV_OUTSIDE_GRID = OUTDIR / "outside_hourly_grid_summary.csv"
CSV_OUTSIDE_CLEAN = OUTDIR / "outside_hourly_clean_summary.csv"

# If you also want to list files in these directories
DIR_OUTSIDE_HOURLY = ROOT / "analysis" / "data_processed" / "outside_hourly"
DIR_OUTSIDE_CLEAN = ROOT / "analysis" / "data_processed" / "outside_hourly_clean"

# Wavelet window used in your scripts (adjust if your project changes)
WINDOW_LABEL = "2021 hourly window (project window)"


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        # We don't want the logger to fail; include the error in the log.
        return pd.DataFrame({"__read_error__": [f"{type(e).__name__}: {e}"]})
    return None


def md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a small dataframe as a markdown table."""
    if df is None:
        return "_(missing)_"
    if "__read_error__" in df.columns:
        return f"_(failed to read: {df['__read_error__'].iloc[0]})_"
    if len(df) == 0:
        return "_(empty)_"
    d = df.copy().head(max_rows)
    return d.to_markdown(index=False)


def safe_listdir(path: Path, pattern: str = "*.csv", max_items: int = 200) -> list[str]:
    if not path.exists():
        return []
    items = sorted([p.name for p in path.glob(pattern)])
    return items[:max_items]


def classify_tiers(clean_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Heuristic tiers for wavelet suitability.
    - Tier A: dense hourly candidates
    - Tier B: OK but sparse
    - Tier C: BROKEN_TIME_1970
    - Tier D: NO_DATA_IN_WINDOW
    """
    tiers = {"Tier A": [], "Tier B": [], "Tier C": [], "Tier D": [], "Other": []}

    if clean_df is None or "__read_error__" in clean_df.columns or "status" not in clean_df.columns:
        return tiers

    for _, r in clean_df.iterrows():
        f = str(r.get("file", ""))
        status = str(r.get("status", ""))
        na_frac = r.get("pm25_na_frac", None)
        step_1h = r.get("step_1h_count", None)
        step_1d = r.get("step_1d_count", None)

        if status == "BROKEN_TIME_1970":
            tiers["Tier C"].append(f)
            continue
        if status == "NO_DATA_IN_WINDOW":
            tiers["Tier D"].append(f)
            continue
        if status != "OK":
            tiers["Other"].append(f)
            continue

        # Dense hourly heuristic:
        # - NA fraction low (<= 0.05) AND
        # - Many 1-hour steps (>= 6000 for a year-ish set) OR small 1-day steps
        is_dense = False
        try:
            if na_frac is not None and float(na_frac) <= 0.05:
                is_dense = True
            # Some of yours are very dense with step_1h ~ 8760
            if step_1h is not None and float(step_1h) >= 6000:
                is_dense = True
            # If daily steps dominate strongly, it's probably sparse
            if step_1d is not None and step_1h is not None:
                if float(step_1d) > 2.0 * max(float(step_1h), 1.0):
                    is_dense = False
        except Exception:
            pass

        if is_dense:
            tiers["Tier A"].append(f)
        else:
            tiers["Tier B"].append(f)

    return tiers


def main() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    out_md = OUTDIR / f"session_log_{today}.md"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df_audit = read_csv_if_exists(CSV_OUTSIDE_AUDIT)
    df_ingest = read_csv_if_exists(CSV_OUTSIDE_INGEST_SUMMARY)
    df_qc = read_csv_if_exists(CSV_QC_OUTSIDE)
    df_grid = read_csv_if_exists(CSV_OUTSIDE_GRID)
    df_clean = read_csv_if_exists(CSV_OUTSIDE_CLEAN)

    # Status counts (clean summary)
    status_counts = None
    if df_clean is not None and "__read_error__" not in df_clean.columns and "status" in df_clean.columns:
        status_counts = df_clean["status"].value_counts().rename_axis("status").reset_index(name="count")

    tiers = classify_tiers(df_clean)

    # A concise “best” ranking table if clean_df exists
    best_ok = None
    if df_clean is not None and "__read_error__" not in df_clean.columns:
        cols = ["file", "status", "pm25_na_frac", "pm25_nonnull", "pm25_negative", "step_1h_count", "step_1d_count", "out_file"]
        have = [c for c in cols if c in df_clean.columns]
        if "status" in df_clean.columns:
            best_ok = (
                df_clean[df_clean["status"] == "OK"][have]
                .sort_values(["pm25_na_frac"], ascending=True)
                .reset_index(drop=True)
            )

    created_files = [
        str(CSV_OUTSIDE_AUDIT),
        str(CSV_OUTSIDE_INGEST_SUMMARY),
        str(CSV_QC_OUTSIDE),
        str(CSV_OUTSIDE_GRID),
        str(CSV_OUTSIDE_CLEAN),
    ]

    outside_hourly_files = safe_listdir(DIR_OUTSIDE_HOURLY)
    outside_clean_files = safe_listdir(DIR_OUTSIDE_CLEAN)

    steps = [
        {
            "Step": 1,
            "Script / Command": "python analysis/src/08_audit_outside_xlsx_deep.py",
            "Goal": "Deep-audit outside XLSX structure",
            "Key output": "analysis/outputs/outside_xlsx_audit_deep.csv",
        },
        {
            "Step": 2,
            "Script / Command": "python analysis/src/10_ingest_outside_xlsx.py",
            "Goal": "Ingest XLSX → standardized outside CSV",
            "Key output": "analysis/data_processed/outside_hourly/outside_*.csv + analysis/outputs/outside_ingest_summary.csv",
        },
        {
            "Step": 3,
            "Script / Command": "python analysis/src/11_qc_outside_hourly_summary.py",
            "Goal": "QC summary (time_min/max, NA, negatives)",
            "Key output": "analysis/outputs/qc_outside_hourly_summary.csv",
        },
        {
            "Step": 4,
            "Script / Command": "python analysis/src/12_make_outside_hourly_grid.py",
            "Goal": "Reindex to full hourly grid (diagnostic)",
            "Key output": "analysis/outputs/outside_hourly_grid_summary.csv + outside_*_grid.csv",
        },
        {
            "Step": 5,
            "Script / Command": "python analysis/src/12b_debug_outside_time_alignment.py",
            "Goal": "Debug time parsing & alignment (hourly vs daily vs broken)",
            "Key output": "console output (kept in terminal history)",
        },
        {
            "Step": 6,
            "Script / Command": "python analysis/src/13_clean_outside_to_hourly_window.py",
            "Goal": f"Clean + clip to {WINDOW_LABEL}; write wavelet-ready files",
            "Key output": "analysis/outputs/outside_hourly_clean_summary.csv + analysis/data_processed/outside_hourly_clean/*.csv",
        },
    ]
    steps_df = pd.DataFrame(steps)

    # Build markdown (no broken tables: each row is one line via to_markdown)
    md = []
    md.append(f"# Session log — {today} (outside-park)")
    md.append("")
    md.append("## What we did (Outside XLSX → CSV → QC → grid → clean window)")
    md.append(md_table(steps_df))
    md.append("")
    md.append("## Key summary files (existence check)")
    md.append("")
    md.append("| Path | Exists |")
    md.append("|---|---:|")
    for p in created_files:
        md.append(f"| `{p}` | {'✅' if Path(p).exists() else '❌'} |")
    md.append("")
    md.append("## Clean summary: status counts")
    md.append(md_table(status_counts) if status_counts is not None else "_(missing or unreadable)_")
    md.append("")
    md.append("## Clean summary: OK files ranked by NA fraction (best first)")
    md.append(md_table(best_ok, max_rows=50) if best_ok is not None else "_(missing or unreadable)_")
    md.append("")
    md.append("## Tier classification (heuristic for wavelet suitability)")
    md.append("")
    md.append("- **Tier A (main wavelet candidates; dense hourly)**")
    md.append("  - " + (", ".join(tiers["Tier A"]) if tiers["Tier A"] else "_none_"))
    md.append("- **Tier B (OK but sparse / daily-heavy; use cautiously / sensitivity)**")
    md.append("  - " + (", ".join(tiers["Tier B"]) if tiers["Tier B"] else "_none_"))
    md.append("- **Tier C (BROKEN_TIME_1970)**")
    md.append("  - " + (", ".join(tiers["Tier C"]) if tiers["Tier C"] else "_none_"))
    md.append("- **Tier D (NO_DATA_IN_WINDOW)**")
    md.append("  - " + (", ".join(tiers["Tier D"]) if tiers["Tier D"] else "_none_"))
    if tiers["Other"]:
        md.append("- **Other statuses**")
        md.append("  - " + ", ".join(tiers["Other"]))
    md.append("")
    md.append("## Raw outside_hourly files present")
    md.append("")
    if outside_hourly_files:
        md.append("```text")
        md.extend(outside_hourly_files)
        md.append("```")
    else:
        md.append("_(folder missing or empty)_")
    md.append("")
    md.append("## Cleaned outside_hourly_clean files present")
    md.append("")
    if outside_clean_files:
        md.append("```text")
        md.extend(outside_clean_files)
        md.append("```")
    else:
        md.append("_(folder missing or empty)_")
    md.append("")
    md.append("## Notes captured from today’s run")
    md.append("")
    md.append("- `openpyxl` warnings about conditional formatting are typically harmless (formatting-only).")
    md.append("- Many `pd.to_datetime` warnings happened because pandas had to guess formats / saw mixed tz; cleaning script should standardize this.")
    md.append("- Some outside files were confirmed as **hourly**, some as **daily-heavy**, and some had **broken 1970 timestamps**.")
    md.append("")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[DONE] wrote {out_md}")


if __name__ == "__main__":
    main()
