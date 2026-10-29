"""Standard task CLI; importing this module never alters plotting backends."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    image,airlight=load_observations(data_path)
    restored,transmission,_=reconstruct(image,airlight)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=restored[None],transmission=transmission[None])
    return restored,transmission

if __name__=='__main__':
    run()
