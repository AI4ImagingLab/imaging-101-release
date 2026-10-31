"""Standard task entry point."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main():
    root=Path(__file__).resolve().parents[1]
    start=time.perf_counter()
    output,_=reconstruct(load_observations(root/'data/raw_data.npz'))
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',**output)
    print(f'Reconstructed four conditions in {time.perf_counter()-start:.3f} s')

if __name__=='__main__': main()
