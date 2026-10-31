"""Standard task entry point, python -m src.main."""
from pathlib import Path
import argparse,time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct


def run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    """Infer fresh fields from observations, save both required NPZ arrays."""
    started=time.perf_counter()
    stress,intensity,info=reconstruct(*load_observations(input_path))
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,reconstruction=stress[None],stress_intensity=intensity[None])
    return stress,intensity,info,time.perf_counter()-started


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw_data.npz');p.add_argument('--output',default='output/reconstruction.npz')
    a=p.parse_args();_,intensity,_,seconds=run(a.input,a.output)
    print(f'Saved {a.output}; K_I,K_II={intensity} MPa sqrt(m); CPU {seconds:.3f}s')


if __name__=='__main__':main()
