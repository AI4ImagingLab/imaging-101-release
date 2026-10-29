"""Standard fresh full-day reconstruction; run from the task workspace."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(raw_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    data,_=load_data(raw_path)
    result=reconstruct(data,progress=True)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,reconstruction=result['reconstruction'])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',default='data/raw_data.npz')
    p.add_argument('--output',default='output/reconstruction.npz');args=p.parse_args()
    result=run(args.raw,args.output)
    print(f"Full-day local reconstruction: {result['runtime_seconds']:.3f} s",flush=True)

if __name__=='__main__':main()
