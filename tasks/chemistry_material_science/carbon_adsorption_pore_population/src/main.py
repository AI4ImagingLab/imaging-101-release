"""Standard complete96-point adsorption inverse."""
from pathlib import Path
import numpy as np,time
from .preprocessing import load_observations
from .solvers import reconstruct
ROOT=Path(__file__).resolve().parents[1]
def run(data_path=None,output_path=None):
    start=time.perf_counter();raw=load_observations(ROOT/'data/raw_data.npz' if data_path is None else data_path);fit=reconstruct(raw)
    p=ROOT/'output/reconstruction.npz' if output_path is None else Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**{k:fit[k][None] for k in ('reconstruction','bin_weights','cumulative_weights')});fit['runtime_seconds']=time.perf_counter()-start;return fit
if __name__=='__main__':
    f=run();print('Full96-point inverse objective',f['objective'],'iterations',f['iterations'])
