"""Reference stereo reconstruction. Run with python -m src.main."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_observations
from .physics_model import cost_volume, depth_from_disparity
from .solvers import aggregate

def reconstruct(raw_path, method='mgm', directions=8, p1=8., p2=32., capture_points=None):
    """Reconstruct the bound full-resolution pair; return output and diagnostics.

    p1/p2 follow source CLI units: multiply by the three RGB input channels.
    Public disparity is minus the ascending native [-127,0] label.
    """
    left, right = load_observations(raw_path)
    cost = cost_volume(left, right)
    labels, diagnostics = aggregate(cost, method, directions, 3*p1, 3*p2, capture_points)
    disparity = (127-labels).astype(np.float32)
    output = {'disparity': disparity[None],
              'reconstruction': depth_from_disparity(disparity).astype(np.float32)[None]}
    return output, diagnostics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw_data.npz'))
    parser.add_argument('--output', type=Path, default=Path('output/reconstruction.npz'))
    args = parser.parse_args()
    output, _ = reconstruct(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)

if __name__ == '__main__':
    main()
