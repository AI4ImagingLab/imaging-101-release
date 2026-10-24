"""Standard entry point; imports never change Matplotlib backend."""
from pathlib import Path
import argparse,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(input_path: str | Path='data/raw_data.npz', output_path: str | Path='output/reconstruction.npz') -> dict[str,np.ndarray]:
    """Reconstruct all bound measured deflections and write pressure NPZ."""
    result=reconstruct(load_observations(input_path));p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**result);return result

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw_data.npz');p.add_argument('--output',default='output/reconstruction.npz');args=p.parse_args();run(args.input,args.output)

if __name__=='__main__':main()
