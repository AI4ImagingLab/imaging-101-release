"""Standard complete four-kernel cosmological acceleration reconstruction."""
from pathlib import Path
import time,json,numpy as np
from .solvers import reconstruct

def run(output_path='output/reconstruction.npz'):
    start=time.perf_counter();result,records=reconstruct()
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**result)
    print(json.dumps({'seconds':time.perf_counter()-start,'hubble_constants':result['hubble_constant'].tolist(),'records':records},default=lambda x:x.tolist()))
    return result

if __name__=='__main__':run()
