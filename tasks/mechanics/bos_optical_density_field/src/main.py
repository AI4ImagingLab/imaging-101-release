"""Standard end-to-end reconstruction: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1]
    result=reconstruct(load_observations(root/'data/raw_data.npz'))
    target=root/'output/reconstruction.npz';target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,reconstruction=result['reconstruction'],density_sigma=result['density_sigma'])
    print('Wrote',target)
if __name__=='__main__':main()
