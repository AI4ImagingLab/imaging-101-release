"""Standard entry point; imports do not change Matplotlib's backend."""
import json,time
from .preprocessing import ROOT,load_observations
from .solvers import reconstruct
import numpy as np

def main():
    start=time.perf_counter();result=reconstruct(load_observations())
    (ROOT/'output').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'output/reconstruction.npz',**result)
    print(json.dumps({'seconds':time.perf_counter()-start,'shape':list(result['reconstruction'].shape),'dtype':'float64'}))
    return result

if __name__=='__main__':main()
