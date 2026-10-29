"""Standard entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    image, psf = load_data(root / 'data/raw_data.npz')
    result = reconstruct(image, psf)
    (root / 'output').mkdir(exist_ok=True)
    np.savez_compressed(root / 'output/reconstruction.npz', reconstruction=result[None])

if __name__ == '__main__':
    main()
