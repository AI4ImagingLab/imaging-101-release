"""Standard entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct


def main():
    """Reconstruct bound observations and write output/reconstruction.npz."""
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / 'data/raw_data.npz')
    result = reconstruct(data['signal'], data['bvalues'][0], data['directions'][0])
    (root / 'output').mkdir(exist_ok=True)
    np.savez_compressed(root / 'output/reconstruction.npz', reconstruction=result)
    return result


if __name__ == '__main__':
    main()
