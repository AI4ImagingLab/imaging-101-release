"""Standard complete reconstruction path; import does not mutate backend."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct


def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    output,diag=reconstruct(load_data(data_path));path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**output)
    return output,diag


if __name__=='__main__':
    output,diag=run();print({k:v['iterations'] for k,v in diag.items()})
