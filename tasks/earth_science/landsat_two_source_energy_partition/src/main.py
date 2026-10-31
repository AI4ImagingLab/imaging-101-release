from pathlib import Path
import time,json,numpy as np
from .solvers import reconstruct
def run(output_path='output/reconstruction.npz'):
    start=time.perf_counter();result,diagnostics=reconstruct();path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**result)
    print(json.dumps({'seconds':time.perf_counter()-start,'canopy_flags':{str(k):int(v) for k,v in zip(*np.unique(diagnostics['pt_flag'],return_counts=True))},'bare_flags':diagnostics['soil_flag'].tolist()}))
    return result
if __name__=='__main__':run()
