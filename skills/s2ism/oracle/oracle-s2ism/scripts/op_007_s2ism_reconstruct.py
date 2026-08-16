#!/usr/bin/env python
"""Standalone public-case s2ISM ML-EM reconstruction helper.
Run from the evaluation run root with: python ${CLAUDE_SKILL_DIR}/scripts/op_007_s2ism_reconstruct.py
It writes outputs/case_000.npz.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import rfftn, irfftn, ifftshift


def _pad_psf_to_image(h, ny, nx):
    # h is torch tensor (Nz, hy, hx, Nch). F.pad order is last dim backwards.
    dy = ny - h.shape[1]
    dx = nx - h.shape[2]
    if dy < 0 or dx < 0:
        raise ValueError('PSF is larger than image')
    if dy or dx:
        py0 = dy // 2
        py1 = dy - py0
        px0 = dx // 2
        px1 = dx - px0
        h = F.pad(h, (0, 0, px0, px1, py0, py1, 0, 0), mode='constant', value=0.0)
    return h


def _prepare_psf(psf, ny, nx, device):
    h = torch.as_tensor(psf, dtype=torch.float32, device=device)
    h = _pad_psf_to_image(h, ny, nx)
    h = h / h.sum(dim=(1, 2, 3), keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
    h_adj = torch.flip(h, dims=(1, 2, 3))
    H = rfftn(h, s=(ny, nx), dim=(1, 2))
    Ha = rfftn(h_adj, s=(ny, nx), dim=(1, 2))
    return H, Ha


def _forward(H, x, ny, nx):
    Xf = rfftn(x, s=(ny, nx), dim=(1, 2))
    yhat_f = (H * Xf[..., None]).sum(dim=0)
    yhat = irfftn(yhat_f, s=(ny, nx), dim=(0, 1)).real
    return ifftshift(yhat, dim=(0, 1))


def _backward(Ha, ratio, ny, nx):
    Rf = rfftn(ratio, s=(ny, nx), dim=(0, 1))
    back_f = (Ha * Rf[None, ...]).sum(dim=3)
    back = irfftn(back_f, s=(ny, nx), dim=(1, 2)).real
    return ifftshift(back, dim=(1, 2))


def reconstruct(y, psf, max_iter=50, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        return _reconstruct_on_device(y, psf, max_iter, device)
    except RuntimeError as exc:
        if device == 'cuda' and 'out of memory' in str(exc).lower():
            torch.cuda.empty_cache()
            return _reconstruct_on_device(y, psf, max_iter, 'cpu')
        raise


def _reconstruct_on_device(y, psf, max_iter, device):
    Y = torch.as_tensor(y, dtype=torch.float32, device=device)
    ny, nx, _ = Y.shape
    nz = psf.shape[0]
    H, Ha = _prepare_psf(psf, ny, nx, device)
    X = torch.ones((nz, ny, nx), dtype=torch.float32, device=device) * (Y.sum() / (nz * ny * nx))
    eps = torch.finfo(torch.float32).eps
    for _ in range(max_iter):
        yhat = _forward(H, X, ny, nx)
        ratio = torch.where(yhat < eps, torch.zeros_like(Y), Y / yhat)
        upd = _backward(Ha, ratio, ny, nx)
        X = X * upd
    return X.detach().cpu().numpy()


def main():
    raw = np.load('public/visible_data/cases/case_000/input_data/raw_data.npz')
    y = raw['measurements'][0].astype(np.float32)
    psf = raw['psf'][0].astype(np.float32)
    recon = reconstruct(y, psf, max_iter=50)
    Path('outputs').mkdir(exist_ok=True)
    np.savez('outputs/case_000.npz', reconstruction=recon.astype(np.float32))
    print('wrote outputs/case_000.npz', recon.shape, recon.dtype, float(np.nanmin(recon)), float(np.nanmax(recon)))


if __name__ == '__main__':
    main()
