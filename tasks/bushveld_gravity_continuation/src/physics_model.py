"""Harmonic equivalent-source representation, extracted from Harmonica.
Copyright (c) 2018 The Harmonica Developers. BSD-3-Clause; see test/validation.md, consolidated original licenses.
"""
import numpy as np

def source_coordinates(stations, depth=6000.0):
    """Place a mathematical source depth metres below each station; shape (N,3)."""
    points=np.asarray(stations,dtype=np.float64).copy()
    points[:,2]-=depth
    return points

def greens_matrix(coordinates, sources):
    """Return inverse Euclidean distances (M,N), in inverse metres."""
    coordinates=np.asarray(coordinates,dtype=np.float64).reshape(-1,3)
    sources=np.asarray(sources,dtype=np.float64).reshape(-1,3)
    delta=coordinates[:,None,:]-sources[None,:,:]
    distance=np.sqrt(np.sum(delta*delta,axis=-1))
    if np.any(distance==0):
        raise ValueError('Evaluation point coincides with equivalent source')
    return 1.0/distance

def forward(coordinates,sources,coefficients):
    """Evaluate harmonic field in mGal, preserving coordinate leading dimensions."""
    return (greens_matrix(coordinates,sources)@coefficients).reshape(coordinates.shape[:-1])
