"""Standard CPU entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main():
    root=Path(__file__).resolve().parents[1]
    stations,data,grid=load_observations(root/'data/raw_data.npz')
    image,_,_=reconstruct(stations,data,grid)
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=image[None])

if __name__=='__main__':main()
