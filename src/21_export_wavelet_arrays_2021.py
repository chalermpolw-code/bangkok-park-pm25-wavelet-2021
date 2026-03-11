#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import pandas as pd
import pywt

IN_DIR = "analysis/data_processed/pairs_2021"
OUT_DIR = "analysis/data_processed/wavelet_arrays_2021"
os.makedirs(OUT_DIR, exist_ok=True)

DT_HOURS = 1.0
PERIOD_MIN_H = 2
PERIOD_MAX_H = 14 * 24
N_PERIODS = 60

MAX_GAP_H = 3
TRIM_H = 24
TZ = "Asia/Bangkok"

SMOOTH_TIME_H = 24
SMOOTH_SCALE_BINS = 3
EPS = 1e-12

def interpolate_small_gaps(x, max_gap):
    x = x.copy()
    n = len(x)
    isn = np.isnan(x)
    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < n and isn[j]:
            j += 1
        L = j - i
        if L <= max_gap:
            left = i - 1
            right = j
            if left >= 0 and right < n and (not np.isnan(x[left])) and (not np.isnan(x[right])):
                x[i:j] = np.linspace(x[left], x[right], L + 2)[1:-1]
        i = j
    return x

def make_periods():
    return np.exp(np.linspace(np.log(PERIOD_MIN_H), np.log(PERIOD_MAX_H), N_PERIODS))

def prep(x):
    x = x.astype(float)
    if TRIM_H > 0 and len(x) > 2 * TRIM_H:
        x = x[TRIM_H:-TRIM_H]
    x = interpolate_small_gaps(x, MAX_GAP_H)
    x = x - np.nanmean(x)
    mask = np.isnan(x)
    x0 = x.copy()
    x0[mask] = 0.0
    return x0, mask

def cwt(x0, periods_h):
    wavelet = "morl"
    fc = pywt.central_frequency(wavelet)
    scales = fc * (periods_h / DT_HOURS)
    W, _ = pywt.cwt(x0, scales, wavelet, sampling_period=DT_HOURS)
    return W

def smooth_time_same(mat, win):
    if win <= 1:
        return mat
    if win % 2 == 0:
        win += 1
    pad = win // 2
    k = np.ones(win) / win
    mp = np.pad(mat, ((0,0),(pad,pad)), mode="reflect")
    out = np.empty_like(mat, dtype=mat.dtype)
    for i in range(mat.shape[0]):
        out[i,:] = np.convolve(mp[i,:], k, mode="valid")
    return out

def smooth_scale_same(mat, win):
    if win <= 1:
        return mat
    if win % 2 == 0:
        win += 1
    pad = win // 2
    k = np.ones(win) / win
    mp = np.pad(mat, ((pad,pad),(0,0)), mode="reflect")
    out = np.empty_like(mat, dtype=mat.dtype)
    for j in range(mat.shape[1]):
        out[:,j] = np.convolve(mp[:,j], k, mode="valid")
    return out

def smooth_2d_same(mat, tw, sw):
    out = smooth_time_same(mat, tw)
    out = smooth_scale_same(out, sw)
    return out

def coherence(Wx, Wy, valid2d):
    Wxy = (Wx * np.conj(Wy)).astype(np.complex128)
    Sxx = (np.abs(Wx)**2).astype(float)
    Syy = (np.abs(Wy)**2).astype(float)

    Wxy[~valid2d] = 0.0 + 0.0j
    Sxx[~valid2d] = 0.0
    Syy[~valid2d] = 0.0
    w = valid2d.astype(float)

    tw = int(max(1, round(SMOOTH_TIME_H / DT_HOURS)))
    sw = int(max(1, SMOOTH_SCALE_BINS))

    num_re = smooth_2d_same(Wxy.real, tw, sw)
    num_im = smooth_2d_same(Wxy.imag, tw, sw)
    den_x  = smooth_2d_same(Sxx, tw, sw)
    den_y  = smooth_2d_same(Syy, tw, sw)
    ww     = smooth_2d_same(w,   tw, sw)

    num_re = np.divide(num_re, ww, out=np.zeros_like(num_re), where=(ww>0))
    num_im = np.divide(num_im, ww, out=np.zeros_like(num_im), where=(ww>0))
    den_x  = np.divide(den_x,  ww, out=np.zeros_like(den_x),  where=(ww>0))
    den_y  = np.divide(den_y,  ww, out=np.zeros_like(den_y),  where=(ww>0))

    num = num_re + 1j*num_im
    R2 = (np.abs(num)**2) / (den_x*den_y + EPS)
    R2 = np.clip(R2, 0, 1)

    # NEW: Calculate Phase Angle
    phase = np.angle(num)

    R2[~valid2d] = np.nan
    phase[~valid2d] = np.nan

    return R2, phase

def main():
    periods_h = make_periods()
    files = sorted(glob.glob(os.path.join(IN_DIR, "park*_outside*_2021.csv")))
    if not files:
        raise SystemExit("No pair files found.")

    for fp in files:
        base = os.path.basename(fp).replace(".csv","")
        df = pd.read_csv(fp)

        xin = pd.to_numeric(df["pm25_in"], errors="coerce").to_numpy()
        xout= pd.to_numeric(df["pm25_out"], errors="coerce").to_numpy()

        xin0, mask_in = prep(xin)
        xout0, mask_out = prep(xout)

        Wx = cwt(xin0, periods_h)
        Wy = cwt(xout0, periods_h)

        valid_t = ~(mask_in | mask_out)
        valid_t = valid_t[:Wx.shape[1]]
        valid2d = np.tile(valid_t, (Wx.shape[0], 1))

        Pin = np.abs(Wx)**2
        Pout= np.abs(Wy)**2

        ratio = np.divide(Pin, Pout, out=np.full_like(Pin, np.nan), where=(Pout>0))
        Att = np.full_like(Pin, np.nan, dtype=float)
        Att[valid2d] = 10.0*np.log10(ratio[valid2d])

        # Receive R2 and phase
        R2, phase = coherence(Wx, Wy, valid2d)

        # NEW: Cone of Influence (COI) calculation for Morlet
        N = xin0.shape[0]
        t_edge = np.minimum(np.arange(N), N - 1 - np.arange(N)) * DT_HOURS
        coi_h = t_edge * np.sqrt(2) # Standard Morlet e-folding derived limit

        out_npz = os.path.join(OUT_DIR, f"{base}.npz")
        np.savez_compressed(
            out_npz,
            pair=base,
            periods_h=periods_h,
            att_db=Att,
            coh_r2=R2,
            phase=phase, # Saved for future use
            pin=Pin,
            pout=Pout,
            coi_h=coi_h, # Saved to plot the shaded cone
            valid=valid2d
        )
        print("[OK] wrote", out_npz)

if __name__ == "__main__":
    main()