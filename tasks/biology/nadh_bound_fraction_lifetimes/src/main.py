"""Complete measured benchmark reconstruction; no plotting side effects."""
from pathlib import Path
import argparse
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    start=time.perf_counter();signal,reference=load_data(input_path)
    result,_=reconstruct(signal,reference)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**result)
    return time.perf_counter()-start

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='data/raw_data.npz');parser.add_argument('--output',default='output/reconstruction.npz')
    args=parser.parse_args();print('Full measured reconstruction: %.3f s'%run(args.input,args.output))

if __name__=='__main__':main()
