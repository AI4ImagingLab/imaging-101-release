"""CPU entry point. Importing this module does not configure Matplotlib."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    raw=load_observations(root/'data/raw_data.npz');result,diagnostics=reconstruct(raw)
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=result)
    return result,diagnostics

if __name__=='__main__':
    result,_=run();print(result)
