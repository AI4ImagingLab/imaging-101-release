"""Standard entry point; importing this module does not alter plot backend."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    """Reconstruct all three repeats and save required NPZ; return submission."""
    result,_=reconstruct(*load_data(input_path))
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**result)
    return result
if __name__=='__main__':run()
