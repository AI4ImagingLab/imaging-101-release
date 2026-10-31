"""Run with python -m src.main from the task directory."""
from pathlib import Path
import time,json,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    """Load public recording, reconstruct and write the single declared NPZ."""
    obs=load_observations(data_path)
    start=time.perf_counter()
    reconstruction=reconstruct(obs['intensity'],obs['extinction'],obs['distances'])
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=reconstruction)
    return reconstruction,time.perf_counter()-start

if __name__=='__main__':
    _,elapsed=run();print(json.dumps({'runtime_seconds':elapsed}))
