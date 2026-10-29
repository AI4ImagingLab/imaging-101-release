"""Standard CLI; python -m src.main from the task workspace."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
from pathlib import Path
import numpy as np
from .solvers import reconstruct

def main():
    with np.load('data/raw_data.npz',allow_pickle=False) as data:
        result=reconstruct(data)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=result['reconstruction'],diffusivity=result['diffusivity'])
    print({'diffusivity_um2_per_s':float(result['diffusivity'][0,0]),'adjusted_r2':result['r2'],'nfev':result['nfev']})

if __name__=='__main__': main()
