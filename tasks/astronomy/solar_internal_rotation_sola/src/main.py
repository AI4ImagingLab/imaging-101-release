"""Standard complete solar rotation reconstruction entry point."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz',progress=True,iterations=1500):
    start=time.perf_counter();data=load_data(data_path)
    output,diagnostics=reconstruct(data,iterations=iterations,progress=progress)
    destination=Path(output_path);destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**output)
    return output,diagnostics,time.perf_counter()-start

if __name__=='__main__':
    _,_,runtime=run()
    print(f'Full converged conditional SOLA inversion completed in {runtime:.6f} s')
