#!/usr/bin/env python
"""Cheap self-check for microscope restoration outputs. Reads only run-local files.

Usage:
  python scripts/op_130_psf_residual_probe.py outputs/case_000.npz \
      public/visible_data/cases/case_000/input_data/raw_data.npz
"""
import sys
import numpy as np
from scipy.ndimage import convolve, laplace


def psf_residual(deconvolved, noisy, psf):
    blurred = convolve(deconvolved.astype(np.float64), psf.astype(np.float64), mode='reflect')
    nz = noisy.astype(np.float64)
    return float(np.mean((blurred - nz) ** 2) / (np.mean(nz ** 2) + 1e-10))


def bg_std(img, block=32, step=16):
    blocks = sorted((img[i:i+16, j:j+16].mean(), i, j)
                    for i in range(0, img.shape[0]-16, step)
                    for j in range(0, img.shape[1]-16, step))
    bi, bj = blocks[0][1], blocks[0][2]
    return float(img[bi:bi+block, bj:bj+block].std())


def main():
    out_path, raw_path = sys.argv[1], sys.argv[2]
    out = np.load(out_path)
    denoised = out['denoised'].astype(np.float64)
    deconvolved = out['deconvolved'].astype(np.float64)
    raw = np.load(raw_path)
    noisy = raw['measurements'][0].astype(np.float64)
    psf = raw['psf'][0].astype(np.float64)

    assert denoised.shape == (502, 502), denoised.shape
    assert deconvolved.shape == (502, 502), deconvolved.shape
    assert np.isfinite(denoised).all() and np.isfinite(deconvolved).all(), 'non-finite output'

    print('PSF residual (deconvolved):', round(psf_residual(deconvolved, noisy, psf), 6))
    print('bg_std noisy / denoised   :', round(bg_std(noisy) / (bg_std(denoised) + 1e-6), 2))
    print('sharpness denoised (lap var):', round(float(laplace(denoised).var()), 3))
    print('sharpness deconvolved       :', round(float(laplace(deconvolved).var()), 3))


if __name__ == '__main__':
    main()
