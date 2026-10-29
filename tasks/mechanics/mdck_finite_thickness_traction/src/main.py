"""Standard task entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    """Actively invert public displacement and save both traction components."""
    traction=reconstruct(load_data())
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=traction[None])
    print(f'Saved traction {traction.shape} Pa')

if __name__=='__main__':main()
