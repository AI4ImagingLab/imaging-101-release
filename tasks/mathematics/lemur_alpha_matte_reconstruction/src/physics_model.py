"""Local color-line matting Laplacian, MIT-derived from PyMatting.

Original PyMatting contributors' MIT copyright is preserved in NOTICE/LICENSE.
No upstream package is required. Window covariance/cofactor conventions follow
the independently captured original implementation.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def mixing_model(alpha, foreground, background):
    """Composite RGB values from dimensionless opacity and two RGB fields."""
    return alpha[..., None] * foreground + (1 - alpha[..., None]) * background


def matting_laplacian(image, known=None, epsilon=1e-7, radius=1):
    """Return CSR color-line Schur complement on native row-major pixels.

    Only windows wholly inside the image are used. Wholly constrained windows
    may be skipped because they cannot change any unknown-row equation.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.float64:
        raise ValueError('image must be a float64 H by W by 3 array')
    if not np.isfinite(image).all() or epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError('finite image and positive finite epsilon required')
    if radius != 1:
        raise ValueError('this selected task supports the published 3 by 3 window')
    h, w, _ = image.shape
    if min(h, w) < 3:
        raise ValueError('at least three pixels per axis required')
    if known is None:
        known = np.zeros((h, w), bool)
    if np.shape(known) != (h, w):
        raise ValueError('known mask shape mismatch')
    all_known = np.ones((h - 2, w - 2), bool)
    for dy in range(3):
        for dx in range(3):
            all_known &= known[dy:dy + h - 2, dx:dx + w - 2]
    yy, xx = np.nonzero(~all_known)
    if not len(yy):
        return coo_matrix((h * w, h * w), dtype=float).tocsr()
    ids = np.stack([(yy + dy) * w + xx + dx for dy in range(3) for dx in range(3)], axis=1)
    colors = image.reshape(-1, 3)[ids]
    means = np.zeros((len(yy), 3))
    for j in range(9):
        means += colors[:, j]
    centered = colors - (means / 9)[:, None]
    a00 = np.full(len(yy), epsilon); a01 = np.zeros(len(yy)); a02 = a01.copy()
    a11 = a00.copy(); a12 = a01.copy(); a22 = a00.copy()
    for j in range(9):
        x, y, z = centered[:, j].T
        a00 += x*x; a01 += x*y; a02 += x*z
        a11 += y*y; a12 += y*z; a22 += z*z
    a00 /= 9; a01 /= 9; a02 /= 9; a11 /= 9; a12 /= 9; a22 /= 9
    determinant = a00*a12*a12+a01*a01*a22+a02*a02*a11-a00*a11*a22-2*a01*a02*a12
    inverse = 1 / determinant
    m00 = (a12*a12-a11*a22)*inverse
    m01 = (a01*a22-a02*a12)*inverse
    m02 = (a02*a11-a01*a12)*inverse
    m11 = (a02*a02-a00*a22)*inverse
    m12 = (a00*a12-a01*a02)*inverse
    m22 = (a01*a01-a00*a11)*inverse
    s, t, u = centered.transpose(2, 0, 1)
    c0 = m00[:, None]*s+m01[:, None]*t+m02[:, None]*u
    c1 = m01[:, None]*s+m11[:, None]*t+m12[:, None]*u
    c2 = m02[:, None]*s+m12[:, None]*t+m22[:, None]*u
    product = c0[:, :, None]*s[:, None, :]+c1[:, :, None]*t[:, None, :]+c2[:, :, None]*u[:, None, :]
    values = np.eye(9)[None]-(1+product)/9
    # Preserve source window accumulation and fixed 25-neighbor CSR slot order.
    # Canonical COO summation is mathematically equivalent, but its ~1e-15
    # reordering changes late finite-precision CG conjugacy on this ill-conditioned
    # system. No convergence threshold is changed to accommodate that variation.
    offsets = np.array([(dy, dx) for dy in range(3) for dx in range(3)])
    slots = (offsets[None, :, 0]-offsets[:, None, 0]+2)*5 + offsets[None, :, 1]-offsets[:, None, 1]+2
    addresses = ids[:, :, None]*25+slots[None]
    data = np.zeros(h*w*25)
    np.add.at(data, addresses.ravel(), values.ravel())
    neighbors = np.array([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)])
    pixels = np.arange(h*w)
    rows = pixels[:, None]//w+neighbors[None, :, 0]
    cols = pixels[:, None]%w+neighbors[None, :, 1]
    indices = np.where((rows >= 0)&(rows < h)&(cols >= 0)&(cols < w), rows*w+cols, 0)
    return csr_matrix((data, indices.ravel(), np.arange(h*w+1, dtype=np.int64)*25), shape=(h*w, h*w))
