"""Standard entry point. Importing this module does not alter plotting backends."""
from pathlib import Path
import argparse,time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
ROOT=Path(__file__).resolve().parents[1]

def run(input_path=None,output_path='output/reconstruction.npz'):
    """Actively reconstruct ALL states and save the declared single output array."""
    start=time.perf_counter();raw=load_data(input_path or ROOT/'data/raw_data.npz')
    result=reconstruct(raw);output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,reconstruction=result['reconstruction'][None])
    return {'runtime_seconds':time.perf_counter()-start,'forward_l2':result['relative_forward_residual'],'output':str(output)}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',default=None);p.add_argument('--output',default='output/reconstruction.npz');a=p.parse_args(argv)
    print(run(a.input,a.output));return 0
if __name__=='__main__':raise SystemExit(main())
