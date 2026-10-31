"""Observed triangular membrane geometry; no force calibration is loaded.

Adapted from foambryo, copyright Sacha Ichbiah and contributors, GPL-3.0-or-later.
See LICENSE and NOTICE for attribution and the exact archived source revision.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray
    region_labels: np.ndarray
    original_vertex_indices: np.ndarray

def load_observations(path):
    """Load batch-one public NPZ, returning unbatched geometry arrays."""
    with np.load(Path(path),allow_pickle=False) as d:
        keys=('vertices','triangles','region_labels')
        if set(d.files)!=set(keys):raise ValueError('Unexpected observation keys')
        result={k:d[k].copy() for k in keys}
    for k,a in result.items():
        if a.ndim!=3 or a.shape[0]!=1:raise ValueError(f'{k}: expected batch-first rank-three array')
        if not np.isfinite(a).all():raise ValueError(f'{k}: non-finite value')
    return {k:a[0] for k,a in result.items()}

def prepare_mesh(vertices,triangles,region_labels):
    """Sort region labels with matching winding reversal; remove unused vertices.

    Coordinates retain author voxel units. Right-hand normals point from the
    first listed material into the second. No triangle or edge is removed.
    """
    vertices=np.asarray(vertices,dtype=np.float64)
    triangles=np.asarray(triangles)
    labels=np.asarray(region_labels)
    if vertices.ndim!=2 or vertices.shape[1]!=3:raise ValueError('vertices must be (V,3)')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or labels.shape!=(len(triangles),2):raise ValueError('Invalid triangle/label dimensions')
    if triangles.dtype.kind not in 'iu' or labels.dtype.kind not in 'iu':raise ValueError('Indices and labels must be integers')
    if triangles.min()<0 or triangles.max()>=len(vertices):raise ValueError('Vertex index outside observed geometry')
    if np.any(labels[:,0]==labels[:,1]):raise ValueError('A membrane must separate two regions')
    if not np.isfinite(vertices).all():raise ValueError('Non-finite coordinates')
    t=triangles.astype(np.int64,copy=True);labels=labels.astype(np.int64,copy=True)
    flip=labels[:,0]>labels[:,1]
    labels[flip]=labels[flip,::-1]
    t[flip]=t[flip][:,[0,2,1]]
    used,inverse=np.unique(t,return_inverse=True)
    return Mesh(vertices[used].copy(),inverse.reshape(t.shape),labels,used)
