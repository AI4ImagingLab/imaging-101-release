"""Standard benchmark entry point; importing this module leaves plotting untouched."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import solve

def main():
    result=solve(load_data())
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=result['reconstruction'],source_depth=result['source_depth'])
    return result

if __name__=='__main__':
    main()
