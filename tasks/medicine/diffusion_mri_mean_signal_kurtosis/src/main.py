"""Standard CLI; importing this module does not select a plotting backend."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    """Reconstruct exact observations, write output/reconstruction.npz."""
    root=Path(__file__).resolve().parents[1]
    result=reconstruct(load_data(root/'data/raw_data.npz'));(root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',**result)
    return result

if __name__=='__main__':main()
