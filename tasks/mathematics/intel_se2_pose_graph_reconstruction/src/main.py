from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    poses,history=reconstruct(load_data());Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=poses[None])
    print(f'Gauss-Newton 20 steps; weighted error {history[0]:.12g} -> {history[-1]:.12g}')

if __name__=='__main__': main()
