"""CPU reference entry point: full selected posterior sampling budget."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import json,time,numpy as np
from .preprocessing import load_data
from .solvers import solve
def main():
    root=Path(__file__).resolve().parents[1];start=time.perf_counter()
    result=solve(load_data(root/'data/raw_data.npz'),progress=lambda n,a:print(f'update={n} mean_acceptance={a:.4f}',flush=True))
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=result['parameters'][None,:])
    print(json.dumps({'total_chi_squared':float(result['chi_squared'].sum()),'acceptance_mean':float(result['acceptance'].mean()),'seconds':time.perf_counter()-start}))
if __name__=='__main__':main()
