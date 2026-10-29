"""Standard complete bound reconstruction entry point."""
from pathlib import Path
import argparse,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(input_path: str | Path='data/raw_data.npz',output_path: str | Path='output/reconstruction.npz') -> dict:
    result=reconstruct(load_observations(input_path));path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw_data.npz');p.add_argument('--output',default='output/reconstruction.npz');a=p.parse_args();run(a.input,a.output)

if __name__=='__main__':main()
