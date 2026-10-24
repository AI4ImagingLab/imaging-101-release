"""Standard CLI; importing this module never selects a Matplotlib backend."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(input_path: str | Path = 'data/raw_data.npz', output_path: str | Path = 'output/reconstruction.npz') -> dict[str,np.ndarray]:
    """Load public observations, reconstruct, write the three required arrays."""
    result=reconstruct(load_observations(input_path))
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**result)
    return result

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    args=parser.parse_args();run(args.input,args.output)

if __name__ == '__main__':
    main()
