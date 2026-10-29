"""Standard execution: python -m src.main."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/raw_data.npz');parser.add_argument('--output',default='output/reconstruction.npz')
    args=parser.parse_args(argv);q,_=reconstruct(load_data(args.data));path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=q[None])
    print(f'Saved {path}; incident flux range {q.min():.9g} to {q.max():.9g} kW/m^2')
    return 0

if __name__=='__main__':raise SystemExit(main())
