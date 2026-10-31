"""Standard raw-data reconstruction; importing does not select a backend."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct


def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    result,diag=reconstruct(load_data(data_path))
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**result)
    return result,diag


if __name__=='__main__':
    result,diag=run();print({k:v for k,v in diag.items() if k!='history'})
