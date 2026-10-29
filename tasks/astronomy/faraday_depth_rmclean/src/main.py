"""Run all observed channels through local RM synthesis and CLEAN."""
from pathlib import Path
import argparse,json
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    observations=load_observations(data_path)
    submission,details=reconstruct(**observations)
    target=Path(output_path);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**submission)
    return submission,details

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/raw_data.npz');parser.add_argument('--output',default='output/reconstruction.npz')
    args=parser.parse_args(argv);_,details=run(args.data,args.output)
    print(json.dumps({key:details[key] for key in ('iterations','cutoff','runtime_seconds')}))
    return 0

if __name__=='__main__':raise SystemExit(main())
