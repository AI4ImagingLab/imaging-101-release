from pathlib import Path
import time,json,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    start=time.perf_counter()
    with load_observations(data_path) as observations:result=reconstruct(observations)
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,reconstruction=result)
    return result,time.perf_counter()-start
if __name__=='__main__':
    result,elapsed=run();print(json.dumps({'shape':list(result.shape),'runtime_seconds':elapsed,'maximum_delta_units':float(result.max())}))
