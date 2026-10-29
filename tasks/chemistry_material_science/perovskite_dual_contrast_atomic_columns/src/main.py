"""Standard active reconstruction entry point."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_data,PIXEL_NM
from .solvers import reconstruct
def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    arrays,_=reconstruct(**load_data(data_path),pixel_nm=PIXEL_NM)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    return arrays
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/raw_data.npz');p.add_argument('--output',default='output/reconstruction.npz');a=p.parse_args()
    run(a.data,a.output)
if __name__=='__main__':main()
