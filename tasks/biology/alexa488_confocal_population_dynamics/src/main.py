from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    t,g=load_observations(root/'data/raw_data.npz')
    p,m,sse=reconstruct(t,g)
    (root/'output').mkdir(exist_ok=True)
    np.savez(root/'output/reconstruction.npz',reconstruction=p[None],correlation=m[None])
    return p,m,sse

if __name__=='__main__':
    print(run())
