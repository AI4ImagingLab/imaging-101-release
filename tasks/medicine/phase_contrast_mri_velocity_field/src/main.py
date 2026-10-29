"""Standard entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import ROOT,load_observations
from .solvers import reconstruct

def main():
    result,_=reconstruct(load_observations())
    path=ROOT/'output/reconstruction.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**result)
    return path

if __name__=='__main__':print(main())
