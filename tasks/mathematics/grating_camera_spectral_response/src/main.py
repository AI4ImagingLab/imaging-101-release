"""Standard independent entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    root=Path(__file__).resolve().parents[1]
    output,_=reconstruct(load_data())
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=output)
    print('Saved all four camera spectral responses',output.shape)

if __name__=='__main__':main()
