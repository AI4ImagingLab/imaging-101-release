"""Standard active reconstruction: python -m src.main."""
from pathlib import Path
import argparse,time
import numpy as np
from .preprocessing import load_raw
from .solvers import reconstruct

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='data/raw_data.npz');parser.add_argument('--output',default='output/reconstruction.npz');args=parser.parse_args(argv)
    started=time.perf_counter();result=reconstruct(load_raw(args.input))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=result['reconstruction'])
    print(f'Complete3166node restoration stress; seconds={time.perf_counter()-started:.6f}; free-force-relative={result["free_dof_force_relative"]:.9g}; energy_Nmm={result["strain_energy_n_mm"]:.9g}',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
