#!/usr/bin/env python
"""Standalone reprojection-NRMSE diagnostic for a lensless reconstruction.

Usage:
    python scripts/op_012_reprojection_probe.py <raw_data.npz> <reconstruction.npz>

Re-applies the FFT forward model A = crop o (h *) to the reconstruction and
reports NRMSE against the measurement. Diagnostic only; mutates nothing.
"""
import sys
import numpy as np
from scipy.fft import next_fast_len, rfft2, irfft2, ifftshift


def build_ops(psf, norm="backward"):
    psf_shape = np.array(psf.shape[:2])
    padded_shape = np.array([next_fast_len(int(s)) for s in (2 * psf_shape - 1)])
    start = (padded_shape - psf_shape) // 2
    end = start + psf_shape

    def pad(v):
        ph, pw = padded_shape
        out = np.zeros((ph, pw, v.shape[2]), dtype=v.dtype)
        out[start[0]:end[0], start[1]:end[1], :] = v
        return out

    def crop(x):
        return x[start[0]:end[0], start[1]:end[1], :]

    H_fft = rfft2(pad(psf), axes=(0, 1), norm=norm)

    def convolve(x_padded):
        Fx = rfft2(x_padded, axes=(0, 1), norm=norm)
        out = irfft2(H_fft * Fx, axes=(0, 1), s=tuple(padded_shape), norm=norm)
        return ifftshift(out, axes=(0, 1))

    def forward(v):
        return crop(convolve(pad(v)))

    return forward


def main():
    raw_path, rec_path = sys.argv[1], sys.argv[2]
    z = np.load(raw_path)
    psf = z["psf"][0].astype(np.float32)
    measurement = z["measurement"][0].astype(np.float32)
    rec = np.load(rec_path)["reconstruction"].astype(np.float32)
    forward = build_ops(psf)
    reproj = forward(rec)
    denom = measurement.max() - measurement.min() + 1e-12
    nrmse = float(np.sqrt(np.mean((reproj - measurement) ** 2)) / denom)
    print("reconstruction shape:", rec.shape, rec.dtype)
    print("finite:", bool(np.isfinite(rec).all()))
    print("reprojection_nrmse:", nrmse)


if __name__ == "__main__":
    main()
