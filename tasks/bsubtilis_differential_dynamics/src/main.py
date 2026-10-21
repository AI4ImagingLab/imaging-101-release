from pathlib import Path
import time,json,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    frames=load_observations(data_path);start=time.perf_counter();parameters,speed=reconstruct(frames)
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,reconstruction=parameters,speed=speed)
    return parameters,speed,time.perf_counter()-start
if __name__=='__main__':
    _,speed,elapsed=run();print(json.dumps({'speed_um_s':speed.tolist(),'runtime_seconds':elapsed}))
