from pathlib import Path
import numpy as np
from .preprocessing import load_data,prepare
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    data=prepare(load_data(data_path));result=reconstruct(data)
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**result)
    return result

if __name__=='__main__':
    run()
