"""Standard CLI. Importing does not select a Matplotlib backend."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    """Write all thermal inverse quantities to one NPZ and return them."""
    result,_=reconstruct(*load_data(input_path));p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**result);return result
if __name__=='__main__':run()
