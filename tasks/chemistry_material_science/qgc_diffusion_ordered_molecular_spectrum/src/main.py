"""Fresh full29-band diffusion inversion; python -m src.main."""
from pathlib import Path
import json,time
import numpy as np
from .preprocessing import prepare
from .solvers import reconstruct

def main():
    start=time.perf_counter();data=prepare();result,details=reconstruct(data)
    Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',**result)
    print(json.dumps({'runtime_s':time.perf_counter()-start,'bands':len(data['groups']),'maximum_ball_ratio':float(np.max(details['residual']/data['eta'])),'minimum_amplitude':float(result['reconstruction'].min())}))
    return result,details

if __name__=='__main__':main()
