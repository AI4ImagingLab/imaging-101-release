"""Compute NUFFT trajectory and pulse-function correction for torchkbnufft.

Self-contained; run with the run-local python.
"""
import numpy as np
import torch


def compute_nufft_params(uv_coords, npix, fov_uas):
    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    fov = fov_uas * uas_to_rad
    psize = fov / npix

    # KbNufft expects (v, u) ordering.
    vu = np.hstack((uv_coords[:, 1:2], uv_coords[:, 0:1]))

    pulsefac_amp = (np.sinc(uv_coords[:, 0] * psize) ** 2
                    * np.sinc(uv_coords[:, 1] * psize) ** 2)
    phase = -np.pi * (uv_coords[:, 0] + uv_coords[:, 1]) * psize
    pulsefac_real = pulsefac_amp * np.cos(phase)
    pulsefac_imag = pulsefac_amp * np.sin(phase)

    vu_scaled = np.array(vu * psize * 2 * np.pi)
    ktraj_vis = torch.tensor(vu_scaled.T, dtype=torch.float32).unsqueeze(0)
    pulsefac_vis = torch.tensor(
        np.stack([pulsefac_real, pulsefac_imag], axis=0), dtype=torch.float32)
    return {'ktraj_vis': ktraj_vis, 'pulsefac_vis': pulsefac_vis}
