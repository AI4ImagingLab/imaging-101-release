"""Standard complete reconstruction entry point; no plotting side effects."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(output=Path('output/reconstruction.npz'),seed=20261029):
    start=time.perf_counter();result,diagnostic=reconstruct(load_data(),seed=seed)
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**result)
    diagnostic['runtime_s']=time.perf_counter()-start
    return result,diagnostic

def main():
    _,info=run()
    print(json.dumps({k:info[k] for k in ['runtime_s','cost','selected_start','successful_starts']},allow_nan=False))

if __name__=='__main__':main()
