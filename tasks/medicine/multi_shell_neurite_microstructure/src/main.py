"""Standard fresh reconstruction entry point; no upstream dependencies."""
from pathlib import Path
import argparse,time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/raw_data.npz'));p.add_argument('--output',type=Path,default=Path('output/reconstruction.npz'));a=p.parse_args(argv)
    data=load_data(a.data);start=time.perf_counter();result,diag=reconstruct(data);a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**result)
    print(dict(seconds=time.perf_counter()-start,voxels=int(data['observation_mask'].sum()),converged=int(diag['converged'].sum()),maximum_iterations=int(diag['iterations'].max())))
    return 0

if __name__=='__main__':raise SystemExit(main())
