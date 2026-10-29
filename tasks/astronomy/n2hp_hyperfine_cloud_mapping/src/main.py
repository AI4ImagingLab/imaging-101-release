"""Standard entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main():
    cube, frequency = load_observations()
    result, _ = reconstruct(cube,frequency)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',**result)
    return result

if __name__ == '__main__':
    main()
