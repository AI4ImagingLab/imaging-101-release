"""python -m src.main writes only the documented submission arrays."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
import time,json,argparse
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw_data.npz');p.add_argument('--output',default='output/reconstruction.npz');a=p.parse_args(argv)
    s,f=load_data(a.input);t=time.perf_counter();result,_=reconstruct(s,f);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**result);print(json.dumps({'seconds':time.perf_counter()-t,'pixels':1024,'output':str(out)}))
if __name__=='__main__':main()
