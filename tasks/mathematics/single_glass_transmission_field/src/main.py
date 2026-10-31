from pathlib import Path
import numpy as np
from .preprocessing import load_observation
from .solvers import reconstruct
def run(root=None):
    root=Path(__file__).resolve().parents[1] if root is None else Path(root)
    y=load_observation(root/'data/raw_data.npz');x=reconstruct(y)
    (root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',reconstruction=x[None])
    return x
if __name__=='__main__':
    x=run();print('Complete fresh reconstruction',x.shape,'unclipped range',x.min(),x.max())
