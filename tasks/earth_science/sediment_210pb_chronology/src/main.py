"""Standard active full-grid reconstruction; no cached-reference mode."""
from pathlib import Path
import time,json,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(output_path=Path('output/reconstruction.npz')):
    start=time.perf_counter();profile,details=reconstruct(*load_observations())
    result={'reconstruction':profile[None,2],'sedimentation_rate':profile[None,1],'initial_activity':profile[None,0],'modeled_activity':profile[None,3]}
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output_path,**result)
    print(json.dumps({'seconds':time.perf_counter()-start,'minimum_chi':float(details['minimum_chi']),'parameters':details['parameters'].tolist(),'confidence_group_sizes':[len(p) for p in details['cloud_parameters']]}))
    return result

if __name__=='__main__':run()
