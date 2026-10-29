"""Standard CPU entry point; importing this module leaves plotting backends unchanged."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    """Load public measurements, reconstruct all quantities, save batch-first NPZ."""
    result=reconstruct(load_data(data_path))
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/raw_data.npz');p.add_argument('--output',default='output/reconstruction.npz');a=p.parse_args();run(a.data,a.output)
