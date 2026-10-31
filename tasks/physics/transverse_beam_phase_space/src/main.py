from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(task_root=None):
    root=Path(task_root) if task_root is not None else Path(__file__).resolve().parents[1]
    result=reconstruct(load_observations(root/'data/raw_data.npz'))
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=result['reconstruction'][None],normalization=result['normalization'][None])
    return result

if __name__=='__main__':run()
