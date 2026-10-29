"""Standard python-m src.main entry point."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    """Read public NPZ and write output/reconstruction.npz; return reconstruction."""
    root=Path(__file__).resolve().parents[1];y,m=load_data(root/'data/raw_data.npz');x=reconstruct(y,m)
    (root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',reconstruction=x[None]);return x
if __name__=='__main__':main()
