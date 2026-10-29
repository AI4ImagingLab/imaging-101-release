"""Explicit finite-volume operators for single-step QSM.

Adapted from QSM.m (c) 2020 kamesy, MIT; full notice and correction
provenance are preserved in test/validation.md. Array axes are x, y, z.
"""
import numpy as np


def wrapped_laplacian(phase, spacing):
    """Wrapped first differences followed by divergence; outer faces are zero."""
    out = np.zeros_like(phase, dtype=np.float64)
    c = (slice(1, -1),) * 3
    for axis, h in enumerate(spacing):
        plus, minus = list(c), list(c)
        plus[axis], minus[axis] = slice(2, None), slice(None, -2)
        df, db = phase[tuple(plus)] - phase[c], phase[c] - phase[tuple(minus)]
        df = np.where((df < -np.pi) | (df > np.pi), (df + np.pi) % (2*np.pi) - np.pi, df)
        db = np.where((db < -np.pi) | (db > np.pi), (db + np.pi) % (2*np.pi) - np.pi, db)
        out[c] += (df - db) / h**2
    return out


def interior_mask(mask):
    """Six-neighbor interior, preserving upstream's center-hole convention."""
    out = np.zeros_like(mask, dtype=bool)
    c = (slice(1, -1),) * 3
    out[c] = True
    for axis in range(3):
        plus, minus = list(c), list(c)
        plus[axis], minus[axis] = slice(2, None), slice(None, -2)
        out[c] &= mask[tuple(plus)] & mask[tuple(minus)]
    return out


def forward_difference(u, axis, spacing):
    """Periodic scratch difference; zero-face states and caller masks remove wrap."""
    return (np.roll(u, -1, axis=axis) - u) / spacing[axis]


def backward_difference(u, axis, spacing):
    """Backward divergence component; negative transpose of forward difference."""
    return (u - np.roll(u, 1, axis=axis)) / spacing[axis]


def laplacian(u, spacing):
    """Centered seven-point Laplacian; masked callers keep outer faces zero."""
    return sum((np.roll(u, -1, axis=a) - 2*u + np.roll(u, 1, axis=a))/h**2
               for a, h in enumerate(spacing))


def wave_operator(u, spacing):
    """W = dzz - Laplacian/3, hence Laplacian(field) = -W(susceptibility)."""
    return sum(c*(np.roll(u, -1, axis=a)-2*u+np.roll(u, 1, axis=a))/h**2
               for a, (h, c) in enumerate(zip(spacing, (-1/3, -1/3, 2/3))))


def symmetric_gradient(w, spacing):
    """Return tensor components xx, xy, xz, yy, yz, zz."""
    d = lambda i, a: forward_difference(w[i], a, spacing)
    return np.array([d(0, 0), (d(1, 0)+d(0, 1))/2,
                     (d(2, 0)+d(0, 2))/2, d(1, 1),
                     (d(2, 1)+d(1, 2))/2, d(2, 2)])


def symmetric_divergence(q, spacing):
    """Negative adjoint of symmetric_gradient in the Frobenius inner product."""
    d = lambda i, a: backward_difference(q[i], a, spacing)
    return np.array([d(0, 0)+d(1, 1)+d(2, 2),
                     d(1, 0)+d(3, 1)+d(4, 2),
                     d(2, 0)+d(4, 1)+d(5, 2)])


def project_balls(vector, radius, tensor=False):
    """Pointwise Euclidean/Frobenius projection; return a fresh array."""
    weights = np.array([1, 2, 2, 1, 2, 1]) if tensor else np.ones(3)
    norm = np.sqrt(np.sum(weights[:, None, None, None]*vector**2, axis=0))
    return vector / np.maximum(1, norm/radius)[None]
