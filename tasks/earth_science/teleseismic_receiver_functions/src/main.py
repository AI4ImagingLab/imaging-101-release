"""Standard entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data, preprocess
from .physics_model import rotate_zne
from .solvers import waterlevel
def reconstruct(data_path='data/raw_data.npz'):
    """Fresh float64 receiver functions (1,2,4801), components R,T."""
    zrt=rotate_zne(preprocess(load_data(data_path)))
    result,_=waterlevel(zrt)
    return result[None,1:]
def main():
    p=Path('output/reconstruction.npz');p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,reconstruction=reconstruct())
    print(p)
if __name__=='__main__':main()
