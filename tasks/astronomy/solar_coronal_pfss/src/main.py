from pathlib import Path
import numpy as np
from .preprocessing import load_data,balance_flux
from .solvers import solve
def reconstruct(path='data/raw_data.npz'):
    """Fresh staggered magnetic fields; batch first, Gauss, float64."""
    return {k:v[None]for k,v in solve(balance_flux(load_data(path))).items()}
def main():
    p=Path('output/reconstruction.npz');p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**reconstruct());print(p)
if __name__=='__main__':main()
