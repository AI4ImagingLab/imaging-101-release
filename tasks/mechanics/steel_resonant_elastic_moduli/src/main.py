"""Standard fresh inverse fit; python -m src.main."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    start=time.perf_counter();observed,weights=load_data()
    result,info=reconstruct(observed,weights)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',**result)
    print(json.dumps({'runtime_s':time.perf_counter()-start,'weighted_rms_percent':info['weighted_rms_percent'],'moduli_GPa':result['reconstruction'].tolist(),'matching_passes':len(info['history'])}))
    return result,info

if __name__=='__main__':main()
