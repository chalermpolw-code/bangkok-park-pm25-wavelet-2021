import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pick_center_time(df, mode="max_diff"):
    """
    mode:
      - max_diff: center on max(pm25_out - pm25_in)
      - max_out:  center on max(pm25_out)
    """
    d = df[["t", "pm25_in", "pm25_out"]].dropna().copy()
    if d.empty:
        raise ValueError("No non-missing rows after dropna().")

    if mode == "max_out":
        idx = d["pm25_out"].idxmax()
    else:
        diff = d["pm25_out"] - d["pm25_in"]
        idx = diff.idxmax()

    return pd.to_datetime(d.loc[idx, "t"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_id", default="park16_outside28_2021",
                    help="e.g., park16_outside28_2021")
    ap.add_argument("--window_days", type=int, default=21,
                    help="window length in days")
    ap.add_argument("--center_mode", default="max_diff",
                    choices=["max_diff", "max_out"],
                    help="how to choose the window center")
    ap.add_argument("--smooth_hours", type=int, default=0,
                    help="optional rolling mean hours (e.g., 3, 6, 24). 0 = no smoothing.")
    args = ap.parse_args()

    infile = f"analysis/data_processed/pairs_2021/{args.pair_id}.csv"
    outdir = "analysis/outputs/landuse_2021/paper_figures"
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(infile):
        raise FileNotFoundError(f"Cannot find: {infile}")

    df = pd.read_csv(infile)

    # Column sanity check
    needed = {"t", "pm25_in", "pm25_out"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(needed)} but found {list(df.columns)}")

    df["t"] = pd.to_datetime(df["t"])
    df = df.sort_values("t").copy()

    # Ensure numeric
    df["pm25_in"] = pd.to_numeric(df["pm25_in"], errors="coerce")
    df["pm25_out"] = pd.to_numeric(df["pm25_out"], errors="coerce")

    # Optional smoothing (helps readability but keep raw for truth)
    if args.smooth_hours and args.smooth_hours > 1:
        df["pm25_in_plot"] = df["pm25_in"].rolling(args.smooth_hours, center=True, min_periods=1).mean()
        df["pm25_out_plot"] = df["pm25_out"].rolling(args.smooth_hours, center=True, min_periods=1).mean()
        y_in = "pm25_in_plot"
        y_out = "pm25_out_plot"
        smooth_tag = f"_smooth{args.smooth_hours}h"
    else:
        y_in = "pm25_in"
        y_out = "pm25_out"
        smooth_tag = ""

    # Choose a defendable window center
    t0 = pick_center_time(df, mode=args.center_mode)
    half = pd.Timedelta(days=args.window_days / 2)
    tmin, tmax = t0 - half, t0 + half

    dz = df[(df["t"] >= tmin) & (df["t"] <= tmax)].copy()

    # Drop rows where either series is missing (for clean shading)
    dz = dz.dropna(subset=[y_in, y_out]).copy()
    if dz.empty:
        raise ValueError("Selected window has no valid overlapping data.")

    # Plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dz["t"], dz[y_out], linewidth=2.0, alpha=0.85, label="Outside (background)")
    ax.plot(dz["t"], dz[y_in], linewidth=2.5, alpha=0.90, label="Inside (park)")

    # Shade only where outside > inside
    mask = dz[y_out].values > dz[y_in].values
    ax.fill_between(
        dz["t"].values,
        dz[y_in].values,
        dz[y_out].values,
        where=mask,
        interpolate=True,
        alpha=0.15,
        label="Outside > Inside (buffered)"
    )

    ax.set_title(
        f"Peak flattening view: {args.pair_id} | center={args.center_mode} | window={args.window_days}d{smooth_tag}",
        pad=12,
        fontweight="bold"
    )
    ax.set_ylabel("PM2.5 (µg/m³)", fontweight="bold")
    ax.set_xlabel("Date", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    out_base = os.path.join(outdir, f"Fig_timeseries_{args.pair_id}_peak_flattening{smooth_tag}")
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)

    print("[OK] wrote", out_base + ".png")
    print("[OK] wrote", out_base + ".pdf")
    print("[INFO] window:", tmin, "to", tmax)

if __name__ == "__main__":
    main()