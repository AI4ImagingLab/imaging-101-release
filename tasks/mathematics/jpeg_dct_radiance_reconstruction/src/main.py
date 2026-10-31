"""Standard task entry point, independent of upstream software."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1]
    a,q=load_observations(root/'data/raw_data.npz')
    u=reconstruct(a,q)
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=u[None])
    return u
if __name__=='__main__':main()
