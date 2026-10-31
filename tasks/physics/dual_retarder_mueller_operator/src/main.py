"""Standard active native-acquisition reconstruction."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    start=time.perf_counter();angles,power=load_data(data_path);out,d=reconstruct(angles,power)
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**out)
    return out,d,time.perf_counter()-start

if __name__=='__main__':
    _,_,runtime=run();print(f'Complete24-angle full-Mueller inverse: {runtime:.6f}s')
