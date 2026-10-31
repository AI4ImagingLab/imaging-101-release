from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def run(task_root=None):
    root=Path(task_root) if task_root is not None else Path(__file__).resolve().parents[1]
    low,core=load_observations(root/'data/raw_data.npz')
    stages=reconstruct(low,core)
    (root/'output').mkdir(exist_ok=True)
    np.savez(root/'output/reconstruction.npz',reconstruction=stages['reconstruction'][None])
    return stages
if __name__=='__main__':run()
