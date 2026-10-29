"""Standard task-local entry point, with no reference dependencies."""
import argparse
from pathlib import Path
import numpy as np
from .preprocessing import load_observations
from .solvers import solve


def run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz', device=None):
    """Load public observations, solve, write all three declared NPZ keys."""
    result,details = solve(load_observations(data_path), device=device)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result,details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    parser.add_argument('--device',choices=['cpu','cuda'])
    args = parser.parse_args()
    _,details = run(args.data,args.output,args.device)
    print({k:v for k,v in details.items() if k not in ['phase_only','raw_rms','snapshots']})


if __name__ == '__main__':
    main()
