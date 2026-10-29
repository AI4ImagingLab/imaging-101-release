"""All real Oysand composite Rayleigh dispersion observations."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
POISSON=np.array([.3,.3,.49,.49])
BOUNDS=np.array([[.0004,.0012],[.0005,.002],[.004,.012],[.08,.16],[.10,.18],[.12,.22],[.15,.28],[.3,.3],[.3,.3],[.49,.49],[.49,.49]])

def load_data(path=None):
    with np.load(path or ROOT/'data/raw_data.npz',allow_pickle=False) as z:return {k:z[k][0].copy() for k in z.files}

def initial_population(rng,popsize=40):
    thickness=rng.uniform(*BOUNDS[:3].T,size=(popsize,3))
    depth=np.column_stack([np.zeros(popsize),thickness.cumsum(axis=1)])
    top=rng.uniform(*BOUNDS[3],size=popsize)
    bottom=[rng.uniform(max(v,BOUNDS[6,0]),BOUNDS[6,1]) for v in top]
    velocity=np.array([np.interp(z,[0.,z[-1]],[a,b]).clip(*BOUNDS[3:7].T) for z,a,b in zip(depth,top,bottom)])
    poisson=rng.uniform(*BOUNDS[7:].T,size=(popsize,4))
    return np.column_stack([thickness,velocity,poisson])
