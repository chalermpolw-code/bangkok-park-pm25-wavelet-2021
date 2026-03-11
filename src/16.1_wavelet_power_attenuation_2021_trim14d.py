#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wavelet power + attenuation (inside vs outside) for 2021 hourly pairs.

Inputs:
  analysis/data_processed/pairs_2021/parkXX_outsideYY_2021.csv
  columns required: t, pm25_in, pm25_out

Outputs:
  analysis/outputs/wavelet_2021_trim14d/
    parkXX_outsideYY_wavelet_summary.csv
    parkXX_outsideYY_attenuation_bandmeans.csv
    figures/parkXX_outsideYY_attenuation_matrix.png
    figures/parkXX_outsideYY_power_inside.png
    figures/parkXX_outsideYY_power_outside.png
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- wavelet backend ----
# Uses PyWavelets. Install if missing:
#   pip install PyWavelets
import pywt

IN_DIR = "analysis/data_processed/pairs_2021"
OUT_DIR = "analysis/outputs/wavelet_2021_trim14d"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DT_COL = "t"
X_IN = "pm25_in"
X_OUT = "pm25_out"

# Sampling: hourly
DT_HOURS = 1.0

# Period grid (hours): 2h to 14 days
PERIOD_MIN_H = 2
PERIOD_MAX_H = 14 * 24
N_PERIODS = 60  # log-spaced

# Gap handling: interpolate very small gaps only
MAX_GAP_H = 3  # only fill gaps up to 3 consecutive hours; bigger gaps stay NaN

# Edge trimming to reduce boundary artifacts (simple COI-lite)
TRIM_H = 14 * 24  # drop first/last 24 hours before wavelet


def interpolate_small_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear-interpolate NaN runs up to max_gap length; keep longer gaps as NaN."""
    x = x.copy()
    n = len(x)
    isn = np.isnan(x)
    if not isn.any():
        return x

    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < n and isn[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap:
            left = i - 1
            right = j
            if left >= 0 and right < n and (not np.isnan(x[left])) and (not np.isnan(x[right])):
                x[i:j] = np.linspace(x[left], x[right], gap_len + 2)[1:-1]
        i = j
    return x


def prep_series(df: pd.DataFrame, col: str) -> np.ndarray:
    """Detrend lightly (remove mean) and handle tiny gaps."""
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    # trim edges (boundary effects)
    if TRIM_H > 0 and len(x) > 2 * TRIM_H:
        x = x[TRIM_H:-TRIM_H]

    # interpolate only tiny gaps
    x = interpolate_small_gaps(x, max_gap=int(MAX_GAP_H))

    # remove mean (ignore NaN)
    m = np.nanmean(x)
    x = x - m
    return x


def make_periods():
    return np.exp(np.linspace(np.log(PERIOD_MIN_H), np.log(PERIOD_MAX_H), N_PERIODS))


def cwt_power(x: np.ndarray, periods_h: np.ndarray):
    """
    Compute CWT power using Morlet wavelet.
    We convert period->scale approximately for Morlet.
    """
    # Fill remaining NaNs with 0 for transform (we keep mask separately)
    mask = np.isnan(x)
    x0 = x.copy()
    x0[mask] = 0.0

    # Morlet central frequency
    wavelet = "morl"
    fc = pywt.central_frequency(wavelet)  # cycles/sample
    # scale (samples) ~ fc * period(samples)
    period_samples = periods_h / DT_HOURS
    scales = fc * period_samples

    coeffs, freqs = pywt.cwt(x0, scales, wavelet, sampling_period=DT_HOURS)
    power = np.abs(coeffs) ** 2  # shape: [n_scales, n_time]
    return power, mask


def band_means(periods_h, att_db):
    """Compute mean attenuation in a few interpretable bands."""
    bands = {
        "2-6h": (2, 6),
        "6-18h": (6, 18),
        "18-30h (diurnal)": (18, 30),
        "2-7d": (2*24, 7*24),
        "7-14d": (7*24, 14*24),
    }
    out = {}
    for name, (a, b) in bands.items():
        idx = (periods_h >= a) & (periods_h <= b)
        if idx.any():
            out[name] = float(np.nanmean(att_db[idx, :]))
        else:
            out[name] = np.nan
    return out


def main():
    periods_h = make_periods()

    summary_rows = []
    band_rows = []

    files = sorted(glob.glob(os.path.join(IN_DIR, "park*_outside*_2021.csv")))
    if not files:
        raise SystemExit(f"No pair files found in {IN_DIR}")

    for fp in files:
        base = os.path.basename(fp).replace(".csv", "")
        df = pd.read_csv(fp)
        df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")

        # trim dataframe to match series trimming
        if TRIM_H > 0 and len(df) > 2 * TRIM_H:
            df_trim = df.iloc[TRIM_H:-TRIM_H].reset_index(drop=True)
        else:
            df_trim = df

        xin = prep_series(df, X_IN)
        xout = prep_series(df, X_OUT)

        # CWT powers
        pin, mask_in = cwt_power(xin, periods_h)
        pout, mask_out = cwt_power(xout, periods_h)

        # Mask where either series was missing (after small-gap fill)
        valid_t = ~(mask_in | mask_out)                 # shape: (n_time,)
        valid = np.tile(valid_t, (pin.shape[0], 1))     # shape: (n_scales, n_time)

        # Attenuation: 10*log10(P_in / P_out)
        att = np.full_like(pin, np.nan, dtype=float)
        ratio = np.divide(pin, pout, out=np.full_like(pin, np.nan), where=(pout > 0))
        att[valid] = 10.0 * np.log10(ratio[valid])

        # Summaries
        overall_att = float(np.nanmean(att))
        frac_valid = float(np.mean(valid))

        summary_rows.append({
            "pair": base,
            "n_time": int(att.shape[1]),
            "period_min_h": PERIOD_MIN_H,
            "period_max_h": PERIOD_MAX_H,
            "valid_fraction": frac_valid,
            "mean_attenuation_db": overall_att,
        })

        bands = band_means(periods_h, att)
        rowb = {"pair": base, **bands}
        band_rows.append(rowb)

        # Quick plots (default colors)
        # Attenuation matrix
        plt.figure(figsize=(10, 4))
        plt.imshow(att, aspect="auto", origin="lower")
        plt.yticks(
            ticks=np.linspace(0, len(periods_h)-1, 8),
            labels=[f"{periods_h[int(i)]:.1f}h" for i in np.linspace(0, len(periods_h)-1, 8)],
        )
        plt.title(f"Attenuation (dB) = 10log10(P_in/P_out) | {base}")
        plt.xlabel("time (trimmed hours)")
        plt.ylabel("period")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{base}_attenuation_matrix.png"), dpi=200)
        plt.close()

        # Inside power
        plt.figure(figsize=(10, 4))
        plt.imshow(np.log10(pin + 1e-12), aspect="auto", origin="lower")
        plt.title(f"log10 Power (inside) | {base}")
        plt.xlabel("time (trimmed hours)")
        plt.ylabel("scale index")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{base}_power_inside.png"), dpi=200)
        plt.close()

        # Outside power
        plt.figure(figsize=(10, 4))
        plt.imshow(np.log10(pout + 1e-12), aspect="auto", origin="lower")
        plt.title(f"log10 Power (outside) | {base}")
        plt.xlabel("time (trimmed hours)")
        plt.ylabel("scale index")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{base}_power_outside.png"), dpi=200)
        plt.close()

        print(f"[OK] wavelet attenuation for {base} | mean_att_db={overall_att:.3f} | valid_frac={frac_valid:.3f}")

    # Write summaries
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "wavelet_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(band_rows).to_csv(os.path.join(OUT_DIR, "attenuation_bandmeans.csv"), index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {os.path.join(OUT_DIR, 'wavelet_summary.csv')}")
    print(f"[DONE] wrote {os.path.join(OUT_DIR, 'attenuation_bandmeans.csv')}")
    print(f"[DONE] figures in {FIG_DIR}")


if __name__ == "__main__":
    main()