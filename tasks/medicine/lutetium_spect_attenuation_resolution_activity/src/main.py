"""CLI: python -m src.main [--device cpu|cuda]."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .preprocessing import load_data,scatter_estimate
from .physics_model import Acquisition
from .solvers import reconstruct


def run(data_path, output_path, device=None):
    torch.set_num_threads(1)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data(data_path)
    acquisition = Acquisition(data['attenuation'],data['angles'],data['radii'],device)
    image,_ = reconstruct(acquisition,data['energy_windows'][0],
                          scatter_estimate(data['energy_windows'],device))
    output = Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,reconstruction=image.cpu().numpy()[None])
    return image


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default=None)
    parser.add_argument('--data',type=Path,default=root/'data/raw_data.npz')
    parser.add_argument('--output',type=Path,default=root/'output/reconstruction.npz')
    args = parser.parse_args()
    run(args.data,args.output,args.device)


if __name__=='__main__':
    main()
