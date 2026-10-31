from pathlib import Path
import time,json
import numpy as np
from .preprocessing import load_raw,prepare
from .solvers import reconstruct

ROOT=Path(__file__).resolve().parents[1]
def run(raw_path=None,output_path=None):
    start=time.perf_counter();data=prepare(load_raw(raw_path or ROOT/'data/raw_data.npz'))
    parameters,curves,diagnostics=reconstruct(data)
    result={'reconstruction':parameters[None,:],'predicted_360':curves['360'][None,:],'predicted_197':curves['197'][None,:]}
    destination=Path(output_path or ROOT/'output/reconstruction.npz');destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**result);diagnostics['seconds']=time.perf_counter()-start
    return result,diagnostics

if __name__=='__main__':
    result,diagnostics=run();print(json.dumps(diagnostics,allow_nan=False))
