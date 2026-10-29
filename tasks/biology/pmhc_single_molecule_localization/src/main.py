"""Canonical entry point: python -m src.main."""
from pathlib import Path
import time,json,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    frames=load_observations(data_path);start=time.perf_counter();table=reconstruct(frames)
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,reconstruction=table)
    return table,time.perf_counter()-start

if __name__=='__main__':
    table,seconds=run();print(json.dumps({'localizations':table.shape[1],'runtime_seconds':seconds}))
