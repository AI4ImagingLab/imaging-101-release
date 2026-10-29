"""Standard task entry point: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    """Read public full radargram and write physically scaled migrated field."""
    data,time,x=load_data();image=reconstruct(data,time)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=image[None])
    print(f'Migrated full profile to {image.shape}')

if __name__=='__main__':main()
