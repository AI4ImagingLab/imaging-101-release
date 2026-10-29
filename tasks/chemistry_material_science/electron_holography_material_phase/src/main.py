"""Run with python -m src.main from the task directory."""
from pathlib import Path
from .preprocessing import load_observations
from .solvers import reconstruct
import numpy as np


def main() -> None:
    data = load_observations()
    result = reconstruct(data['hologram'], data['vacuum'])
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz', reconstruction=result)
    print('Wrote output/reconstruction.npz')


if __name__ == '__main__':
    main()
