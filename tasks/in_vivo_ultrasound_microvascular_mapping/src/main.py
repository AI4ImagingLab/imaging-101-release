from pathlib import Path
import json,time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main():
    """Reconstruct all 6000 frames using radial localization; write submission."""
    start=time.perf_counter();iq=load_observations()
    result=reconstruct(iq,methods=('radial',),progress=lambda i,n:print(f'Block {i}/{n}',flush=True))['radial']
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=result['density'][None],speed=result['speed'][None])
    print(json.dumps({'runtime_seconds':time.perf_counter()-start,'localizations':int(result['counts'][:,0].sum()),'tracks':int(result['counts'][:,1].sum())}))

if __name__=='__main__':main()
