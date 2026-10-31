"""Complete selected CPU reconstruction: python -m src.main."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1]
    result=reconstruct(load_observations(root/'data/raw_data.npz'))
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=result['reconstruction'])
    print(f"Full 576000-curve reconstruction completed in {result['seconds']:.3f} s")
if __name__=='__main__':main()
