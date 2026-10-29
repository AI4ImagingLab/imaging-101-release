"""Run the complete selected reconstruction from public observations."""
from pathlib import Path
import time,numpy as np
from .preprocessing import load_data,ROOT
from .solvers import reconstruct
def run(output=None):
    data=load_data();start=time.perf_counter();x,diagnostics=reconstruct(data['system_matrix'],data['observations'])
    path=Path(output) if output else ROOT/'output/reconstruction.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=x[None]);return x,diagnostics,time.perf_counter()-start
if __name__=='__main__':
    x,info,seconds=run();print(f"{info['iterations']} updates, {seconds:.3f} s")
