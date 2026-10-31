from pathlib import Path
import json
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    theta,info,_=reconstruct(load_data())
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=theta[None])
    print(json.dumps(info,indent=2,allow_nan=False))
if __name__=='__main__':main()
