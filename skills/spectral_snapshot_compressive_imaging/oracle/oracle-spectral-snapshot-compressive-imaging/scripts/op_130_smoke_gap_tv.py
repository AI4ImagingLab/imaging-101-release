"""Short TV-only accelerated GAP smoke test for one case.

Usage:
    public/envs/runtime/.venv-posix/bin/python scripts/op_130_smoke_gap_tv.py <case_input_dir> [n_iter]
where <case_input_dir> contains raw_data.npz. Default n_iter=30.
"""
import os
import sys
import numpy as np
from skimage.restoration import denoise_tv_chambolle


def shift(x, step=1):
    row, col, nC = x.shape
    o = np.zeros((row, col + (nC - 1) * step, nC))
    for i in range(nC):
        o[:, i * step:i * step + col, i] = x[:, :, i]
    return o


def shift_back(x, step=1):
    row, col, nC = x.shape
    for i in range(nC):
        x[:, :, i] = np.roll(x[:, :, i], -step * i, axis=1)
    return x[:, 0:col - step * (nC - 1), :]


def A(x, Phi):
    return np.sum(x * Phi, axis=2)


def At(y, Phi):
    return np.repeat(y[:, :, np.newaxis], Phi.shape[2], axis=2) * Phi


def main(input_dir, n_iter=30):
    raw = np.load(os.path.join(input_dir, 'raw_data.npz'))
    y = raw['measurement'][0].astype(np.float64)
    mask2d = raw['mask'][0].astype(np.float64)
    r, c, nC, step = 256, 256, 31, 1
    Phi = np.zeros((r, c + step * (nC - 1), nC))
    for i in range(nC):
        Phi[:, i:i + c, i] = mask2d
    Phi_sum = np.sum(Phi, axis=2)
    Phi_sum[Phi_sum == 0] = 1
    x = At(y, Phi)
    y1 = np.zeros_like(y)
    lam = 1.0
    for k in range(n_iter):
        yb = A(x, Phi)
        y1 = y1 + (y - yb)
        x = x + lam * At((y1 - yb) / Phi_sum, Phi)
        x = shift_back(x, step=1)
        x = denoise_tv_chambolle(x, 130 / 255.0, max_num_iter=5, channel_axis=-1)
        x = shift(x, step=1)
    cube = shift_back(x, step=1)
    print('smoke cube', cube.shape, 'finite', bool(np.isfinite(cube).all()),
          'range [%.4f, %.4f]' % (float(cube.min()), float(cube.max())))


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else 'public/visible_data/cases/case_000/input_data'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(d, n)
