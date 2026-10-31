"""Full default CPU retrieval. Importing the module does not change plot backend."""
from pathlib import Path
import time,numpy as np
from .preprocessing import load_observations
from .physics_model import TransmissionModel
from .solvers import reconstruct

def run(root=None,seed=20261028,progress=None):
    """Run all50x1000 original stretch transitions, save all declared target arrays."""
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    model=TransmissionModel(load_observations(root/'data/raw_data.npz'));start=time.perf_counter()
    prediction,diagnostics=reconstruct(model,seed=seed,steps=1000,progress=progress)
    runtime=time.perf_counter()-start;(root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',**prediction)
    np.savez_compressed(root/'notebooks/reference_outputs/fresh_measurements.npz',**prediction,**diagnostics,runtime_seconds=np.array(runtime),seed=np.array(seed))
    return prediction,diagnostics,model,runtime

if __name__=='__main__':
    prediction,_,_,runtime=run(progress=lambda n,p:print('step',n,'max log posterior',p,flush=True))
    print('seconds',runtime,'median',prediction['reconstruction'],'best',prediction['best_fit'],flush=True)
