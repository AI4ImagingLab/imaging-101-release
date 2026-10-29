"""Standard entry point."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_frames
from .solvers import solve

def main():
    root=Path(__file__).resolve().parents[1]
    start=time.perf_counter();a,b=load_frames(root/'data/raw_data.npz')
    velocity,confidence=solve(a,b)
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=velocity,confidence=confidence)
    print(f'Reconstructed {velocity.shape} in {time.perf_counter()-start:.6f} s')

if __name__=='__main__':main()
