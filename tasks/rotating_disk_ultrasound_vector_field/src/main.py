"""Standard complete measured-RF reconstruction; no backend changes on import."""
from pathlib import Path
import argparse,time,json
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/raw_data.npz'))
    parser.add_argument('--output',type=Path,default=Path('output/reconstruction.npz'))
    args=parser.parse_args(argv);start=time.perf_counter()
    result=reconstruct(load_observations(args.data))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,reconstruction=result['velocity'][None])
    summary={'runtime_seconds':time.perf_counter()-start,'output':str(args.output),'shape':list(result['velocity'][None].shape),'smoothing':result['smoothing']['history']}
    print(json.dumps(summary));return summary

if __name__=='__main__':main()
