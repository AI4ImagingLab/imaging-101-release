"""Standard public-observation reconstruction entry point."""
from pathlib import Path
import time,json
import numpy as np
from .solvers import reconstruct

def main():
    start=time.perf_counter()
    with np.load('data/raw_data.npz',allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    result=reconstruct(raw);Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=result['reconstruction'])
    print(json.dumps(dict(runtime_seconds=time.perf_counter()-start,best_iteration=result['best_iteration'],trace=result['trace']),indent=2))
if __name__=='__main__':main()
