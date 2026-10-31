"""Run python -m src.main from task workspace; never access hidden references."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def run(task_dir=None):
    root=Path(task_dir) if task_dir is not None else Path(__file__).resolve().parents[1]
    y,angles=load_observations(root/'data/raw_data.npz');result=reconstruct(y,angles)
    output=root/'output/reconstruction.npz';output.parent.mkdir(exist_ok=True);np.savez_compressed(output,**result)
    return result
if __name__=='__main__':run()
