"""Run the complete bound pressure reconstruction."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import solve

def main():
    start=time.perf_counter()
    g,q=load_data();p,history=solve(g,q)
    Path("output").mkdir(exist_ok=True)
    np.savez_compressed("output/reconstruction.npz",reconstruction=p[None])
    print(f"Reconstructed {p.shape} pressure in {time.perf_counter()-start:.6f} s; {len(history)} CG updates")
    return p,history

if __name__=="__main__":main()
