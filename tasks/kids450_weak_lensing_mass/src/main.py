"""Standard task entry point; importing this module never changes a plotting backend."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def run(data_path="data/raw_data.npz",output_path="output/reconstruction.npz"):
    """Actively reconstruct the bound observations and save batch-first physical arrays."""
    data=load_data(data_path)
    arrays=reconstruct(data)
    destination=Path(output_path);destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**arrays)
    return arrays
if __name__=="__main__":
    run()
