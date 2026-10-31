"""Standard CPU reconstruction entry point."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    """Actively reconstruct inputs and write the declared NPZ submission."""
    start=time.perf_counter()
    v,i=load_data(data_path);output,diagnostics=reconstruct(v,i)
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**output)
    return output,diagnostics,time.perf_counter()-start

if __name__=='__main__':
    _,_,seconds=run()
    print(f'Complete native sweep reconstructed in {seconds:.6f} s')
