"""Local one-shot OSMOSIS graph operator. GPL-3.0-or-later; Triad 2024."""
import numpy as np

def spacing(coordinate):
    """Preserve the original VTK float32 first-coordinate difference."""
    return np.float32(coordinate[1] - coordinate[0])

def weights(dx, dy=None):
    dy = dx if dy is None else dy
    return np.float32(float(dy)/(2.0*float(np.float32(dx+dy)))), np.float32(float(dx)/(2.0*float(np.float32(dx+dy))))

def matrix_apply(pressure, dx, dy=None):
    """Apply weighted graph Laplacian, missing boundary edges excluded."""
    p = np.asarray(pressure, dtype=np.float32)
    cx, cy = weights(dx, dy)
    missing_x = np.zeros(p.shape, np.float32); missing_x[:, (0, -1)] = 1
    missing_y = np.zeros(p.shape, np.float32); missing_y[(0, -1), :] = 1
    w = cx*missing_x + cy*missing_y
    out = (p.astype(np.float64)*(1.0-w.astype(np.float64))).astype(np.float32)
    out[:, :-1] -= cx*p[:, 1:]
    out[:, 1:] -= cx*p[:, :-1]
    out[:-1, :] -= cy*p[1:, :]
    out[1:, :] -= cy*p[:-1, :]
    return out

def build_rhs(gradient, dx, dy=None):
    """Trapezoidal gradient edge integrals and boundary-aware divergence."""
    dy = dx if dy is None else dy
    sx, sy = np.asarray(gradient, np.float32)
    cx, cy = weights(dx, dy)
    bx = ((-cx*(sx[:, 1:]+sx[:, :-1])).astype(np.float64)*(float(dx)/2.0))
    by = ((-cy*(sy[1:, :]+sy[:-1, :])).astype(np.float64)*(float(dy)/2.0))
    out = np.zeros_like(sx)
    out[:, :-1] += bx
    out[:, 1:] -= bx
    out[:-1, :] += by
    out[1:, :] -= by
    return out

def edge_residual(pressure, gradient, dx):
    """Return pressure increment minus trapezoidal observed gradient, x/y."""
    p=np.asarray(pressure, np.float64);g=np.asarray(gradient,np.float64)
    return np.diff(p,axis=1)-.5*float(dx)*(g[0,:,1:]+g[0,:,:-1]), np.diff(p,axis=0)-.5*float(dx)*(g[1,1:,:]+g[1,:-1,:])
