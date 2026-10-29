"""Standard fresh movie reconstruction; python -m src.main."""
from pathlib import Path
import json,time
import numpy as np
from .solvers import reconstruct

def main():
    start=time.perf_counter();root=Path(__file__).resolve().parents[1]
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as a:
        result=reconstruct(a['frames'],a['cell_labels'])
    output=root/'output';output.mkdir(exist_ok=True)
    np.savez_compressed(output/'reconstruction.npz',**result)
    print(json.dumps({'seconds':time.perf_counter()-start,'shape':result['reconstruction'].shape,'positive_pixels':int((result['reconstruction']>0).sum())}))

if __name__=='__main__':main()
