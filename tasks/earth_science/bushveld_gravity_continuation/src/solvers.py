"""Explicit column-scaled Tikhonov inversion; no domain solver dependency.
Copyright (c) 2018 The Harmonica Developers. BSD-3-Clause; see test/validation.md, consolidated original licenses.
"""
import numpy as np
from scipy.linalg import solve
from .physics_model import greens_matrix,source_coordinates,forward

DEPTH=6000.0
DAMPING=0.1

def solve_coefficients(kernel,data,damping=DAMPING):
    """Minimize ||K c-d||^2 + damping ||diag(std(K)) c||^2.

    Column standard deviations use population normalization and no centering
    of the kernel itself, matching Verde's StandardScaler(with_mean=False).
    Return coefficients (N,) mGal m and scaling (N,) inverse metres.
    """
    if damping<=0:
        raise ValueError('Damping must be positive')
    scale=np.std(kernel,axis=0,ddof=0)
    if np.any(scale==0):
        raise ValueError('Degenerate kernel column')
    design=kernel/scale
    normal=design.T@design
    normal.flat[::normal.shape[0]+1]+=damping
    scaled=solve(normal,design.T@data,assume_a='pos')
    return scaled/scale,scale

def reconstruct(stations,data,grid,depth=DEPTH,damping=DAMPING):
    """Return (image(H,W) mGal, coefficients(N,) mGal m, station fit(N,) mGal)."""
    sources=source_coordinates(stations,depth)
    kernel=greens_matrix(stations,sources)
    coefficients,_=solve_coefficients(kernel,data,damping)
    return forward(grid,sources,coefficients),coefficients,kernel@coefficients
