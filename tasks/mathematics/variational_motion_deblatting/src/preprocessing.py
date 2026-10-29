"""Observation loading and explicit antialiased trajectory rasterization."""
from pathlib import Path
import numpy as np

OBJECT_SHAPE = (36, 36)


def load_data(path='data/raw_data.npz'):
    """Return BGR observation and independent background, float64 (169,137,3)."""
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = [data[k] for k in ('observation', 'background')]
    for array in arrays:
        if array.shape != (1, 169, 137, 3) or array.dtype != np.float64:
            raise ValueError('Expected float64 arrays of shape (1,169,137,3).')
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite observation.')
    return arrays[0][0], arrays[1][0]


def boundingBox(img, pads=None):
    rows = np.where(np.any(img, axis=1))[0]
    cols = np.where(np.any(img, axis=0))[0]
    rmin, rmax, cmin, cmax = rows[0], rows[-1], cols[0], cols[-1]
    if pads is not None:
        rmin, rmax = max(rmin-pads[0], 0), min(rmax+pads[0], img.shape[0])
        cmin, cmax = max(cmin-pads[1], 0), min(cmax+pads[1], img.shape[1])
    return rmin, rmax, cmin, cmax


def line_aa(r0, c0, r1, c1):
    """Zingl line rasterization, matching scikit-image 0.22 conventions.

    Adapted from scikit-image's BSD-3-Clause _draw.pyx; see test/validation.md, consolidated original licenses.
    """
    rr, cc, val = [], [], []
    dc, dr = abs(c0-c1), abs(r0-r1)
    err = float(dc-dr)
    sign_c, sign_r = (1 if c0<c1 else -1), (1 if r0<r1 else -1)
    ed = np.sqrt(dc*dc+dr*dr) if dc+dr else 1.
    c, r = c0, r0
    while True:
        cc.append(c); rr.append(r); val.append(abs(err-dc+dr)/ed)
        olderr, oldc = err, c
        if 2*olderr >= -dc:
            if c == c1:
                break
            if olderr+dr < ed:
                cc.append(c); rr.append(r+sign_r); val.append(abs(olderr+dr)/ed)
            err -= dr
            c += sign_c
        if 2*olderr <= dr:
            if r == r1:
                break
            if dc-olderr < ed:
                cc.append(oldc+sign_c); rr.append(r); val.append(abs(dc-olderr)/ed)
            err += dc
            r += sign_r
    return np.asarray(rr), np.asarray(cc), 1-np.asarray(val)


def renderTraj(pars, H):
    """Rasterize upstream straight/quadratic paths in row/column coordinates."""
    if pars.shape[1] == 2:
        pars = np.concatenate((pars, np.zeros((2, 1))), 1)
        ns = 2
    else:
        ns = 5
    times = np.linspace(0, 1, ns)
    for a, b in zip(times[:-1], times[1:]):
        start = np.round(pars[:, 0]+pars[:, 1]*a+pars[:, 2]*a*a).astype(int)
        end = np.round(pars[:, 0]+pars[:, 1]*b+pars[:, 2]*b*b).astype(int)
        rr, cc, val = line_aa(*start, *end)
        valid = (rr>0)&(cc>0)&(rr<H.shape[0])&(cc<H.shape[1])
        H[rr[valid], cc[valid]] = val[valid]
    return H
