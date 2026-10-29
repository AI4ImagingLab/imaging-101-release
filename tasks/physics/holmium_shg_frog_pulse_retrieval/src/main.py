"""Standard entry: python -m src.main."""
from pathlib import Path
import json,time,os
if __name__=='__main__':
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from .preprocessing import load_observations,gaussian_initialization,unit_energy_field
from .solvers import Retriever

def reconstruct(data_path,iterations=1000,seed=1234,alpha=.25,width_fs=50.):
    """Return normalized complex field[512], retriever diagnostics, and model."""
    measured,model=load_observations(data_path)
    result=Retriever(model,measured,gaussian_initialization(model.grid,width_fs),alpha=alpha,seed=seed).run(iterations)
    return unit_energy_field(model.grid,result['spectrum']),result,model

def main():
    root=Path(__file__).resolve().parents[1]
    e,result,_=reconstruct(root/'data/raw_data.npz');(root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=e[None])
    print(json.dumps({k:result[k] for k in ['actual_error','runtime_seconds']}))

if __name__=='__main__':main()
