"""Standard CPU entry point; no upstream dependencies."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import prepare_data
from .solvers import reconstruct
ROOT=Path(__file__).resolve().parents[1]

def main():
    start=time.perf_counter()
    fields,_=reconstruct(prepare_data(ROOT/"data/raw_data.npz"))
    (ROOT/"output").mkdir(exist_ok=True)
    np.savez_compressed(ROOT/"output/reconstruction.npz",**fields)
    print(f"Reconstructed428 locations /20 types in {time.perf_counter()-start:.3f}s")
if __name__=='__main__':main()
