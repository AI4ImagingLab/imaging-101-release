"""Standard native-observation entry point."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

ROOT = Path(__file__).resolve().parents[1]


def main(input_path=None, output_path=None):
    """Reconstruct and save one output NPZ; return elapsed seconds and count."""
    input_path = ROOT / 'data/raw_data.npz' if input_path is None else Path(input_path)
    output_path = ROOT / 'output/reconstruction.npz' if output_path is None else Path(output_path)
    before = time.perf_counter()
    phases, _, _ = load_observations(input_path)
    result, support, _ = reconstruct(phases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, reconstruction=result[None])
    return {'seconds': time.perf_counter()-before, 'evaluated_voxels': int(support.sum())}


if __name__ == '__main__':
    print(main())
