from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    result=reconstruct(load_data(data_path))[None]
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,reconstruction=result)
    return result
if __name__=='__main__':run()
