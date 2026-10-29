from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    out=Path('output');out.mkdir(exist_ok=True)
    result,_=reconstruct(load_data());np.savez_compressed(out/'reconstruction.npz',reconstruction=result)
if __name__=='__main__':main()
