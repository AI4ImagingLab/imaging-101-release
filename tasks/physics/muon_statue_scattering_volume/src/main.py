"""Run the full bound CPU reconstruction: python -m src.main."""
from pathlib import Path
import numpy as np
import torch
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    image=reconstruct(load_data(root/'data/raw_data.npz'))
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=image[None])
    return image

if __name__=='__main__':
    main()
