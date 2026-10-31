"""Standard fresh reconstruction entry point; importing this module has no side effects."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main() -> None:
    start=time.perf_counter();frames,times=load_observations();result=reconstruct(frames,times)
    Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',**result)
    print(json.dumps({'runtime_s':time.perf_counter()-start,'output':'output/reconstruction.npz'}))

if __name__=='__main__':main()
