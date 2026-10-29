"""Public acquisition loading and observation-only covariance estimation."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

POISSON_RATIO = 0.35
THICKNESS = 0.05
BOUNDARY_MULTIPLIER = 100000.0
RELATIVE_NOISE = 0.001
GLOBAL_NOISE = 0.001
ADHOC_RELATIVE_WEIGHT = 0.01

@dataclass(frozen=True)
class Acquisition:
    nodes: np.ndarray
    triangles: np.ndarray
    displacement: np.ndarray
    nodal_force: np.ndarray
    fixed_dofs: np.ndarray

def load_data(path: str | Path = 'data/raw_data.npz') -> Acquisition:
    """Load strict batch-one acquisition; return unbatched float64/indices/mask."""
    schema = {'nodes': ((1,441,2),np.dtype('float64')),
              'triangles': ((1,800,3),np.dtype('int64')),
              'displacement': ((1,441,2),np.dtype('float64')),
              'nodal_force': ((1,441,2),np.dtype('float64')),
              'fixed_dofs': ((1,441,2),np.dtype('bool'))}
    values = {}
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files) != set(schema):
            raise ValueError('Unexpected acquisition keys')
        for key,(shape,dtype) in schema.items():
            value=archive[key]
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f'Invalid acquisition array {key}')
            values[key]=value[0].copy()
    if values['triangles'].min()<0 or values['triangles'].max()>=441:
        raise ValueError('Triangle index outside nodes')
    return Acquisition(**values)

def element_incenters(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Triangle incenter coordinates in the acquisition length unit."""
    points=nodes[triangles]
    lengths=np.stack([np.linalg.norm(points[:,1]-points[:,2],axis=1),
                      np.linalg.norm(points[:,0]-points[:,2],axis=1),
                      np.linalg.norm(points[:,0]-points[:,1],axis=1)],axis=1)
    return np.sum(lengths[:,:,None]*points,axis=1)/lengths.sum(axis=1)[:,None]

def observation_weights(displacement: np.ndarray) -> np.ndarray:
    """Ad-hoc inverse covariance from public u_m only; not exact noise variance."""
    measured=np.asarray(displacement,dtype=np.float64).reshape(-1)
    variance=2*((GLOBAL_NOISE*np.ptp(measured))**2+
                (ADHOC_RELATIVE_WEIGHT*np.abs(measured))**2)
    if not np.isfinite(variance).all() or np.any(variance<=0):
        raise ValueError('Nonpositive observation-based covariance')
    return 1/variance
