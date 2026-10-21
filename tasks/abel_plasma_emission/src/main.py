"""Standard execution: python -m src.main, from the task workspace."""
import argparse
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_projection
from .solvers import reconstruct


def run(input_path='data/raw_data.npz', output_path='output/reconstruction.npz'):
    """Actively reconstruct the supplied observations and save one NPZ array."""
    start = time.perf_counter()
    result = reconstruct(load_projection(input_path))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, reconstruction=result)
    return {'runtime_seconds': time.perf_counter()-start, 'shape': result.shape}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/raw_data.npz')
    parser.add_argument('--output', default='output/reconstruction.npz')
    args = parser.parse_args()
    print(run(args.input, args.output))


if __name__ == '__main__':
    main()
