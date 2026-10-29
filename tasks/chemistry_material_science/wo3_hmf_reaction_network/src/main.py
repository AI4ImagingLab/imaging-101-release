"""Standard entry point: python -m src.main."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_data,preprocess
from .solvers import reconstruct

def main():
    start=time.perf_counter()
    c,q,initial=preprocess(load_data())
    result,info=reconstruct(c,q,initial)
    out=Path('output');out.mkdir(exist_ok=True)
    np.savez_compressed(out/'reconstruction.npz',**result)
    info['runtime_s']=time.perf_counter()-start
    print(json.dumps(info))
    return result,info

if __name__=='__main__':main()
