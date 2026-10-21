"""Standard entry point: python -m src.main."""
from pathlib import Path
import json
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(data_path: str | Path='data/raw_data.npz',
        output_path: str | Path='output/reconstruction.npz') -> dict:
    """Actively reconstruct observations and write two real channels in DN."""
    field,info=reconstruct(load_observations(data_path))
    path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=np.stack([field.real,field.imag])[None])
    return info

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
