"""Standard complete100frame reconstruction entry point."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(task_root=None):
    root=Path(task_root) if task_root else Path(__file__).resolve().parents[1]
    start=time.perf_counter()
    result=reconstruct(load_observations(root/"data/raw_data.npz"))
    (root/"output").mkdir(exist_ok=True)
    np.savez_compressed(root/"output/reconstruction.npz",
                        reconstruction=result["flatfield"][None],baseline=result["baseline"][None])
    result["seconds"]=time.perf_counter()-start
    return result

if __name__=="__main__":
    result=run()
    print(json.dumps(dict(seconds=result["seconds"],selected_smoothness=result["smoothness"],search=result["search"],history=result["history"])))
