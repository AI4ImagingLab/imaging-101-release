"""Standard execution writes fresh original L1-IRLS surface normals."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def main():
    start=time.perf_counter();observations,lights,support=load_observations()
    result=reconstruct(observations,lights,support)
    out=Path('output/reconstruction.npz');out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,reconstruction=result[None])
    print(json.dumps({'shape':[1,*result.shape],'runtime_seconds':time.perf_counter()-start}))
    return result
if __name__=='__main__':main()
