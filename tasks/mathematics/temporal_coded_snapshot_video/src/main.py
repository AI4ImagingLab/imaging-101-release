"""Standard full-bound reconstruction: python -m src.main."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(output=Path('output/reconstruction.npz')):
    start=time.perf_counter();result,diagnostics=reconstruct(load_data())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**result)
    return result,diagnostics,time.perf_counter()-start

def main():
    result,diagnostics,elapsed=run()
    print(json.dumps({'runtime_s':elapsed,'shape':list(result['reconstruction'].shape),'method':'GAP-TV100','residuals':[d['relative_observation_residual_history'][-1] for d in diagnostics]},allow_nan=False))

if __name__=='__main__':main()
