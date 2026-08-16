#!/usr/bin/env python
"""Mawet et al. (2014) small-sample t-test SNR for a companion.

Diagnostic only; does not modify submissions. Run with run-local python.
"""
import numpy as np


def _aperture_centers(r_px, pa_deg, fwhm, exclude_nearest=0):
    n = max(1, int(2 * np.pi * r_px / fwhm))
    start = np.deg2rad(pa_deg + 90.0)
    delta = 2 * np.pi / n
    centers = [(r_px * np.cos(start), r_px * np.sin(start))]
    for i in range(1 + exclude_nearest, n - exclude_nearest):
        t = start + i * delta
        centers.append((r_px * np.cos(t), r_px * np.sin(t)))
    return centers


def _aperture_median(image, cx, cy, ox, oy, radius):
    h, w = image.shape
    yy, xx = np.mgrid[:h, :w]
    m = ((xx - (cx + ox)) ** 2 + (yy - (cy + oy)) ** 2) <= radius ** 2
    return float(np.nanmedian(image[m]))


def compute_snr(image, planet_x, planet_y, fwhm, exclude_nearest=1):
    h, w = image.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx, dy = planet_x - cx, planet_y - cy
    r_px = np.hypot(dx, dy)
    pa = float(np.degrees(np.arctan2(dy, dx)) - 90.0)
    ap_r = fwhm / 2.0
    centers = _aperture_centers(r_px, pa, fwhm, exclude_nearest)
    vals = [_aperture_median(image, cx, cy, ox, oy, ap_r) for ox, oy in centers]
    signal = vals[0]
    noise = np.asarray(vals[1:], dtype=np.float64)
    if noise.size < 2:
        return float("nan")
    return float((signal - noise.mean()) /
                 (noise.std(ddof=1) * np.sqrt(1 + 1.0 / noise.size)))


if __name__ == "__main__":
    print("import compute_snr and call on your detection map")
