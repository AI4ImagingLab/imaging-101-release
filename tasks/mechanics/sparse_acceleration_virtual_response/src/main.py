"""Standard full virtual-response reconstruction entry point."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz',callback=None):
    """Learn HP and reconstruct all fields,save three required NPZ keys."""
    fields,info=reconstruct(load_observations(input_path),callback=callback)
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**{k:fields[k][None] for k in ['reconstruction','velocity','acceleration']})
    return fields,info
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw_data.npz');p.add_argument('--output',default='output/reconstruction.npz');a=p.parse_args()
    fields,info=run(a.input,a.output,callback=lambda phase,k,n:print(f'{phase} {k}/{n}',flush=True))
    print(f'Saved {a.output};HP={info["hyperparameters"]};CPU seconds={info["runtime_seconds"]:.3f}')
if __name__=='__main__':main()
