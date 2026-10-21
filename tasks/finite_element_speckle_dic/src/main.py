"""Standard CPU reconstruction entry point."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct


def main():
    displacement, history = reconstruct(*load_data())
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz', reconstruction=displacement[None])
    print(f'Converged in {len(history)} updates; final increment {history[-1,1]:.3g} pixels')


if __name__ == '__main__':
    main()
