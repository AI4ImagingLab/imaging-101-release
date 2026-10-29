from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    lower,upper=load_data(data_path)
    result,history=reconstruct(lower,upper)
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,reconstruction=result[None])
    return result,history

if __name__=='__main__':run()
