"""Run: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    """Read public NPZ and write physical position and magnetic moment arrays."""
    p,m,*_=reconstruct(*load_data())
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',positions=p[None],moments=m[None])
    print(f'Reconstructed {len(p)} magnetic sources')

if __name__=='__main__':main()
