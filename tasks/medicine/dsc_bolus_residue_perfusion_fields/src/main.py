"""Standard entry point: python -m src.main. No plotting/backend import side effects."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import signal_to_concentration
from .solvers import deconvolve,leakage_volume

def reconstruct(raw_path='data/raw_data.npz',output_path='output/reconstruction.npz',comparisons=False):
    """Actively reconstruct all89588 supported voxels/all160lags from raw signal.

    Return primary submission and identified computed diagnostics. Optional
    comparisons additionally compute complete sSVD/cSVD on the same data.
    """
    start=time.perf_counter()
    with np.load(raw_path,allow_pickle=False) as raw:
        signal=raw['signal'][0];support=raw['support'][0];aif=raw['arterial_input'][0]
    concentration,s0,bolus=signal_to_concentration(signal,support)
    volume=leakage_volume(concentration,aif,support,bolus)
    selected=concentration[support];method_results={}
    for method in (['svd','csvd','osvd'] if comparisons else ['osvd']):
        method_results[method]=deconvolve(selected,aif,method=method)
    primary=method_results['osvd'];residue=np.zeros(support.shape+(160,),dtype=np.float64)
    residue[support]=primary['residual'];flow=np.zeros(support.shape);flow[support]=primary['map']
    transit=np.zeros(support.shape);transit[support]=volume['cbv_lc'][support]/flow[support]
    submission=dict(reconstruction=residue[None],relative_flow=flow[None],relative_volume=volume['cbv_lc'][None],transit_time=transit[None])
    if output_path is not None:
        target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**submission)
    return submission,dict(methods=method_results,concentration=concentration,baseline=s0,baseline_counts=bolus,volume=volume,support=support,aif=aif,signal=signal,seconds=time.perf_counter()-start)

if __name__=='__main__':
    import json
    result,diagnostics=reconstruct()
    print(json.dumps(dict(seconds=diagnostics['seconds'],supported_voxels=int(diagnostics['support'].sum()),output='output/reconstruction.npz')))
