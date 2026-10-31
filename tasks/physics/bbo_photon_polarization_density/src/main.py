"""CLI: python -m src.main; paths relative to the task workspace."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    counts,projectors=load_data();rho,intensity,_=reconstruct(counts,projectors)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=rho[None],intensity=np.array([[intensity]],dtype=np.float64))
if __name__=='__main__': main()
