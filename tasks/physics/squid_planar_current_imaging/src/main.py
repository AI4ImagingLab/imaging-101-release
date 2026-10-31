from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    root=Path(__file__).resolve().parents[1]
    result,stage=reconstruct(load_data(root/'data/raw_data.npz'))
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',**result)
    print('Wrote full padded two-component current; imaginary ratio:',stage['imaginary_ratio'])

if __name__=='__main__':main()
