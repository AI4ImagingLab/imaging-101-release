"""Standard CPU reconstruction entry point."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
ROOT=Path(__file__).resolve().parents[1]
def run(data_path=None,output_path=None):
    data_path=ROOT/'data/raw_data.npz' if data_path is None else Path(data_path)
    output_path=ROOT/'output/reconstruction.npz' if output_path is None else Path(output_path)
    result,_=reconstruct(load_data(data_path));output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,**result)
    return result
if __name__=='__main__':run()
