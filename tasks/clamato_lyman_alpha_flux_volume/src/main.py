"""Standard module entry point; importing it has no plotting/backend side effects."""
from pathlib import Path
import json
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(task_dir=None):
    root=Path(task_dir) if task_dir is not None else Path(__file__).resolve().parents[1]
    observations=load_observations(root/'data/raw_data.npz')
    field,alpha,info=reconstruct(observations)
    out=root/'output';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'reconstruction.npz',reconstruction=field[None])
    print(json.dumps({k:v for k,v in info.items() if np.isscalar(v)},indent=2),flush=True)
    return field,alpha,info

if __name__=='__main__':
    run()
