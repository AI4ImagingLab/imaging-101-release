"""Full selected inference entry point; importing never changes plot backend."""
from pathlib import Path
import time,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(root=None,seed=20261028,progress=None):
    """Execute both robust fits, runtime residual map and all100+1000MCMC steps."""
    from .physics_model import SpectralModel
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    model=SpectralModel(load_observations(root/'data/raw_data.npz'));start=time.perf_counter()
    prediction,diagnostics=reconstruct(model,seed=seed,progress=progress);seconds=time.perf_counter()-start
    (root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',**prediction)
    likelihood=diagnostics['likelihood']
    np.savez_compressed(root/'notebooks/reference_outputs/fresh_measurements.npz',**prediction,chain=diagnostics['chain'][None],log_probability=diagnostics['log_probability'][None],acceptance_fraction=diagnostics['acceptance_fraction'][None],initial_fit=likelihood['first_parameters'][None],preliminary=likelihood['parameters'][None],parameter_error=likelihood['parameter_error'][None],noise_map=likelihood['full_error_map'][None],noise_reference=likelihood['errors'][None],runtime_seconds=np.array(seconds),seed=np.array(seed))
    return prediction,diagnostics,model,seconds

if __name__=='__main__':
    result,_,_,seconds=run(progress=lambda phase,n,p:print(phase,n,'best log probability',p,flush=True))
    print('runtime_seconds',seconds,'parameters',result['reconstruction'],flush=True)
