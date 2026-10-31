"""Standard CLI: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    image,spectra=load_data(data_path)
    result,diagnostics=reconstruct(image,spectra)
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,reconstruction=result[None])
    return result,diagnostics

if __name__=='__main__':
    run()
