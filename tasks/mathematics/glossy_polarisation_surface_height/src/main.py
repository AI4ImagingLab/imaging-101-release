"""Standard entry point: python -m src.main."""
from pathlib import Path
import argparse
import time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct


def run(input_path='data/raw_data.npz', output_path='output/reconstruction.npz'):
    """Reconstruct public observations and save float64[1,H,W]."""
    started=time.perf_counter()
    arrays=load_observations(input_path)
    height,diagnostics=reconstruct(*arrays)
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,reconstruction=height[None])
    return height,diagnostics,time.perf_counter()-started


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    args=parser.parse_args()
    _,_,seconds=run(args.input,args.output)
    print(f'Reconstruction saved to {args.output}; CPU runtime {seconds:.3f} s')


if __name__=='__main__':
    main()
