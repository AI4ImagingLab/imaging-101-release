from pathlib import Path
import numpy as np

def load_data(path=Path('data/raw_data.npz')):
    """Load batch-one relative poses (m,m,rad) and information matrices."""
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k][0].copy() for k in ['edge_indices','relative_poses','information']}
    if d['edge_indices'].shape!=(1483,2) or d['relative_poses'].shape!=(1483,3) or d['information'].shape!=(1483,3,3): raise ValueError('Invalid observation shapes')
    if not all(np.all(np.isfinite(v)) for v in d.values()): raise ValueError('Nonfinite observations')
    return d

def initialize(data):
    """Compose all 1227 sequential observed transforms; fix node zero to identity."""
    from .physics_model import wrap
    ids=data['edge_indices'];z=data['relative_poses'];n=int(ids.max())+1;x=np.zeros((n,3))
    for j in range(1,n):
        matches=np.flatnonzero((ids[:,0]==j-1)&(ids[:,1]==j))
        if len(matches)!=1: raise ValueError('Expected exactly one sequential odometry edge')
        v=z[matches[0]];c=np.cos(x[j-1,2]);s=np.sin(x[j-1,2])
        x[j,:2]=x[j-1,:2]+[c*v[0]-s*v[1],s*v[0]+c*v[1]]
        x[j,2]=wrap(x[j-1,2]+v[2])
    return x
