"""Standard entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .solvers import reconstruct
def main():
    """Reconstruct bound observations and write output/reconstruction.npz."""
    root=Path(__file__).resolve().parents[1]
    result=reconstruct(root/'data/raw_data.npz')
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',**result)
if __name__=='__main__': main()
