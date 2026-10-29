"""Standard task entry point, with paths relative to task directory."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    images,psf=load_data('data/raw_data.npz')
    output,_=reconstruct(images,psf)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',**output)

if __name__=='__main__':
    main()
