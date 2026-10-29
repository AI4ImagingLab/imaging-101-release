"""Standard CPU entry point."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    result=reconstruct(load_data());Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',**result)
    return result
if __name__=='__main__': main()
