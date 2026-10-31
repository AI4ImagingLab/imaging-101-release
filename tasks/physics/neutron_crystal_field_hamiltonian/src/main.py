"""Run complete active Voigt + three-stage CEF reconstruction."""
from pathlib import Path
import time,json
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1];start=time.perf_counter()
    predictions,diagnostics=reconstruct(load_data(root/'data/raw_data.npz'));elapsed=time.perf_counter()-start
    (root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',**predictions)
    arrays=dict(predictions,voigt_parameters=diagnostics['voigt_parameters'],voigt_covariance=diagnostics['voigt_covariance'],width_covariance=diagnostics['width_covariance'],initial_parameters=diagnostics['initial_parameters'],warmup_parameters=diagnostics['warmup_parameters'],runtime_seconds=np.array(elapsed))
    for i in range(3):arrays[f'trace{i}_x']=diagnostics['trace_x'][i];arrays[f'trace{i}_loss']=diagnostics['trace_loss'][i]
    np.savez_compressed(root/'notebooks/reference_outputs/fresh_measurements.npz',**arrays)
    (root/'notebooks/reference_outputs/execution.json').write_text(json.dumps(dict(runtime_seconds=elapsed,stages=diagnostics['stages'],selection=diagnostics['selection']),indent=2),encoding='utf8')
    return predictions,diagnostics,elapsed

if __name__=='__main__':
    import os
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    prediction,diagnostics,elapsed=run();print(json.dumps(dict(runtime_seconds=elapsed,stages=diagnostics['stages'],selected_model=int(prediction['selected_model'][0])),indent=2))
