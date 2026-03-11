#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wavelet coherence (inside vs outside) for 2021 hourly pairs
TEST RUN using complex Morlet wavelet.

Inputs:
  analysis/data_processed/pairs_2021/parkXX_outsideYY_2021.csv
  columns: t, pm25_in, pm25_out

Outputs:
  analysis/outputs/wavelet_2021_cmor_test/
    coherence_summary.csv
    coherence_bandmeans.csv
    figures/*_coherence_matrix.png

Dependencies:
  pip install PyWavelets
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

IN_DIR = "analysis/data_processed/pairs_2021"
OUT_DIR = "analysis/outputs/wavelet_2021_cmor_test"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DT_COL = "t"
X_IN = "pm25_in"
X_OUT = "pm25_out"

# hourly sampling
DT_HOURS = 1.0

# periods (hours)
PERIOD_MIN_H = 2
PERIOD_MAX_H = 14 * 24
N_PERIODS = 60

# gap handling
MAX_GAP_H = 3   # interpolate only tiny gaps
TRIM_H = 24     # keep same as current main coherence script

# coherence smoothing (keep unchanged for fair comparison)
SMOOTH_TIME_H = 24
SMOOTH_SCALE_BINS = 3

EPS = 1e-12

# complex Morlet test choice
WAVELET = "cmor1.5-1.0"


def interpolate_small_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear-interpolate NaN runs up to max_gap; keep longer gaps as NaN."""
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


def prep_series(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x: mean-removed series (NaNs allowed)
      mask: True where NaN remains (after tiny gap interpolation)
    """
    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    # trim edges
    if TRIM_H > 0 and len(x) > 2 * TRIM_H:
        x = x[TRIM_H:-TRIM_H]

    # fill only tiny gaps
    x = interpolate_small_gaps(x, max_gap=int(MAX_GAP_H))

    # mean remove (ignore NaN)
    x = x - np.nanmean(x)

    mask = np.isnan(x)
    return x, mask


def make_periods() -> np.ndarray:
    return np.exp(np.linspace(np.log(PERIOD_MIN_H), np.log(PERIOD_MAX_H), N_PERIODS))


def cwt(x: np.ndarray, periods_h: np.ndarray) -> np.ndarray:
    """Return complex CWT coefficients W (n_scales, n_time) using complex Morlet."""
    x0 = x.copy()
    x0[np.isnan(x0)] = 0.0

    wavelet = WAVELET
    fc = pywt.central_frequency(wavelet)
    period_samples = periods_h / DT_HOURS
    scales = fc * period_samples

    W, freqs = pywt.cwt(x0, scales, wavelet, sampling_period=DT_HOURS)
    return W


def smooth_time_same(mat: np.ndarray, win: int) -> np.ndarray:
    """
    SAME-length moving average along time axis (axis=1) with reflect padding.
    Guarantees output shape == input shape.
    """
    if win <= 1:
        return mat
    if win % 2 == 0:
        win += 1

    pad = win // 2
    kernel = np.ones(win, dtype=float) / win

    mp = np.pad(mat, ((0, 0), (pad, pad)), mode="reflect")

    out = np.empty_like(mat, dtype=float if np.isrealobj(mat) else complex)
    for i in range(mat.shape[0]):
        out[i, :] = np.convolve(mp[i, :], kernel, mode="valid")
    return out


def smooth_scale_same(mat: np.ndarray, win: int) -> np.ndarray:
    """
    SAME-length moving average along scale axis (axis=0) with reflect padding.
    Guarantees output shape == input shape.
    """
    if win <= 1:
        return mat
    if win % 2 == 0:
        win += 1

    pad = win // 2
    kernel = np.ones(win, dtype=float) / win

    mp = np.pad(mat, ((pad, pad), (0, 0)), mode="reflect")

    out = np.empty_like(mat, dtype=float if np.isrealobj(mat) else complex)
    for j in range(mat.shape[1]):
        out[:, j] = np.convolve(mp[:, j], kernel, mode="valid")
    return out


def smooth_2d_same(mat: np.ndarray, time_win: int, scale_win: int) -> np.ndarray:
    """Separable smoothing that preserves exact shape."""
    out = mat
    out = smooth_time_same(out, time_win)
    out = smooth_scale_same(out, scale_win)
    return out


def coherence_from_cwts(Wx: np.ndarray, Wy: np.ndarray, valid_mask_2d: np.ndarray) -> np.ndarray:
    """
    Compute smoothed wavelet coherence R^2(s,t).

    valid_mask_2d: True where both series are valid (n_scales, n_time)
    """
    n_scales, n_time = Wx.shape
    valid_mask_2d = valid_mask_2d[:, :n_time]

    Wxy = (Wx * np.conj(Wy)).astype(np.complex128)
    Sxx = (np.abs(Wx) ** 2).astype(float)
    Syy = (np.abs(Wy) ** 2).astype(float)

    Wxy[~valid_mask_2d] = 0.0 + 0.0j
    Sxx[~valid_mask_2d] = 0.0
    Syy[~valid_mask_2d] = 0.0

    w = valid_mask_2d.astype(float)

    time_win = int(max(1, round(SMOOTH_TIME_H / DT_HOURS)))
    scale_win = int(max(1, SMOOTH_SCALE_BINS))

    num_re = smooth_2d_same(Wxy.real, time_win, scale_win)
    num_im = smooth_2d_same(Wxy.imag, time_win, scale_win)
    den_x = smooth_2d_same(Sxx, time_win, scale_win)
    den_y = smooth_2d_same(Syy, time_win, scale_win)
    ww = smooth_2d_same(w, time_win, scale_win)

    num_re = np.divide(num_re, ww, out=np.zeros_like(num_re), where=(ww > 0))
    num_im = np.divide(num_im, ww, out=np.zeros_like(num_im), where=(ww > 0))
    den_x = np.divide(den_x, ww, out=np.zeros_like(den_x), where=(ww > 0))
    den_y = np.divide(den_y, ww, out=np.zeros_like(den_y), where=(ww > 0))

    num = num_re + 1j * num_im

    R2 = (np.abs(num) ** 2) / (den_x * den_y + EPS)
    R2 = np.clip(R2, 0.0, 1.0)

    R2[~valid_mask_2d] = np.nan
    return R2


def band_means(periods_h: np.ndarray, R2: np.ndarray) -> dict:
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
        out[name] = float(np.nanmean(R2[idx, :])) if idx.any() else np.nan
    return out


def main():
    periods_h = make_periods()
    files = sorted(glob.glob(os.path.join(IN_DIR, "park*_outside*_2021.csv")))
    if not files:
        raise SystemExit(f"No pair files found in {IN_DIR}")

    summary_rows = []
    band_rows = []

    for fp in files:
        base = os.path.basename(fp).replace(".csv", "")
        df = pd.read_csv(fp)

        xin, mask_in = prep_series(df, X_IN)
        xout, mask_out = prep_series(df, X_OUT)

        Wx = cwt(xin, periods_h)
        Wy = cwt(xout, periods_h)

        valid_t = ~(mask_in | mask_out)
        valid_t = valid_t[:Wx.shape[1]]
        valid = np.tile(valid_t, (Wx.shape[0], 1))

        R2 = coherence_from_cwts(Wx, Wy, valid)

        mean_R2 = float(np.nanmean(R2))
        frac_valid = float(np.mean(valid))

        summary_rows.append({
            "pair": base,
            "wavelet": WAVELET,
            "n_scales": int(R2.shape[0]),
            "n_time": int(R2.shape[1]),
            "valid_fraction": frac_valid,
            "mean_coherence_R2": mean_R2,
            "smooth_time_h": SMOOTH_TIME_H,
            "smooth_scale_bins": SMOOTH_SCALE_BINS,
        })

        bands = band_means(periods_h, R2)
        band_rows.append({"pair": base, "wavelet": WAVELET, **bands})

        plt.figure(figsize=(10, 4))
        plt.imshow(R2, aspect="auto", origin="lower", vmin=0, vmax=1)
        plt.yticks(
            ticks=np.linspace(0, len(periods_h)-1, 8),
            labels=[f"{periods_h[int(i)]:.1f}h" for i in np.linspace(0, len(periods_h)-1, 8)],
        )
        plt.title(f"Wavelet coherence R² | {base} | {WAVELET}")
        plt.xlabel("time (trimmed hours)")
        plt.ylabel("period")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"{base}_coherence_matrix_{WAVELET.replace('.', '_')}.png"), dpi=200)
        plt.close()

        print(f"[OK] coherence for {base} | wavelet={WAVELET} | mean_R2={mean_R2:.3f} | valid_frac={frac_valid:.3f}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "coherence_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(band_rows).to_csv(os.path.join(OUT_DIR, "coherence_bandmeans.csv"), index=False, encoding="utf-8-sig")

    print(f"[DONE] wrote {os.path.join(OUT_DIR, 'coherence_summary.csv')}")
    print(f"[DONE] wrote {os.path.join(OUT_DIR, 'coherence_bandmeans.csv')}")
    print(f"[DONE] figures in {FIG_DIR}")


if __name__ == "__main__":
    main()