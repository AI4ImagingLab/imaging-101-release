"""Visible acquisition loading; no inferred fields or cached normal systems."""
from pathlib import Path
import numpy as np
KEYS=('coordinates','triangles','displacements','reactions','boundary_masks','free_dofs','load_steps','shape_gradients','element_areas','query_gradients')
def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:
        d={k:z[k][0].copy() for k in KEYS}
    if d['displacements'].shape!=(600,383,2): raise ValueError('Expected complete600-frame acquisition')
    if np.any(np.linalg.det(element_deformation(d['triangles'],d['shape_gradients'],d['displacements'][0]))<=0): raise ValueError('Nonpositive deformation determinant')
    return d
def element_deformation(triangles,shape_gradients,displacement):
    return np.eye(2)+np.einsum('eni,enj->eij',displacement[triangles],shape_gradients)
