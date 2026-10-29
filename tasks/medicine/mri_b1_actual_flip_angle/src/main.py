"""Run python -m src.main to write the required submission."""
from pathlib import Path
import numpy as np
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1];(root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',**reconstruct(root/'data/raw_data.npz'))
if __name__=='__main__':main()
