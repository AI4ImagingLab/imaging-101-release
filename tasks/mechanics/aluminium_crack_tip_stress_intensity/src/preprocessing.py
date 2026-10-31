"""Displacement sampling adapted from DLR CrackPy; see LICENSE/provenance."""
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator

CRACK_TIP = np.array([15.39,.40])
CRACK_ANGLE_DEG = -.27
MIN_RADIUS = 15.39/20
MAX_RADIUS = 15.39/5
GRID_INCREMENT = .01
ANGLE_GAP_DEG = 20.


def load_observations(path):
    """Return coordinates[N,2] and measured displacements[N,2], both mm."""
    with np.load(Path(path),allow_pickle=False) as z:
        x=z['coordinates'][0].copy();u=z['displacements'][0].copy()
    if x.ndim!=2 or x.shape[1]!=2 or u.shape!=x.shape:
        raise ValueError('Expected matching point-by-component coordinate/displacement arrays')
    if not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError('Nonfinite observations')
    return x,u


def transform(coordinates, displacements, tip=CRACK_TIP, angle_deg=CRACK_ANGLE_DEG):
    """Translate the measured tip to zero, rotate positions and vectors alike."""
    a=np.deg2rad(angle_deg)
    rotation=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    return (coordinates-np.asarray(tip))@rotation.T,displacements@rotation.T


def fitting_grid():
    """Return r,phi,x,y, each float64[231,559]; length mm, angle rad."""
    gap=ANGLE_GAP_DEG/180*np.pi
    r,phi=np.mgrid[MIN_RADIUS:MAX_RADIUS:GRID_INCREMENT,-np.pi+gap:np.pi-gap:GRID_INCREMENT]
    return r,phi,r*np.cos(phi),r*np.sin(phi)


def sample_displacements(coordinates,displacements,grid_x,grid_y):
    """Piecewise linear Delaunay interpolation, rebased at the crack tip.

    Outside the convex hull, values remain NaN and are explicitly excluded
    from the fitting equations. No missing observation is replaced by zero.
    """
    interpolator=LinearNDInterpolator(coordinates,displacements)
    tip=interpolator(np.array([[0.,0.]]))[0]
    sampled=interpolator(np.column_stack((grid_x.ravel(),grid_y.ravel())))-tip
    return sampled[:,0].reshape(grid_x.shape),sampled[:,1].reshape(grid_y.shape),tip
