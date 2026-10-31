from pathlib import Path
import time,json,numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    start=time.perf_counter();velocity,observed=load_data();result,records=reconstruct(velocity,observed)
    Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',**result)
    print(json.dumps({'runtime_s':time.perf_counter()-start,'four_source_sse':[r['sse'] for r in records]}))
    return result,records

if __name__=='__main__':main()
