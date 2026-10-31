"""Standard CLI: reconstruct public acquisition and write the batch-one NPZ."""
import argparse
import json
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import prepare_inverse,reconstruct

def run(data_path: str | Path='data/raw_data.npz',
        output_path: str | Path='output/reconstruction.npz',prior: str='tv',
        verbose: bool=False):
    context=prepare_inverse(load_data(data_path))
    result=reconstruct(context,prior=prior,verbose=verbose)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reconstruction=result.reconstruction[None,:])
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    parser.add_argument('--prior',choices=['tv','smooth','tikhonov'],default='tv')
    parser.add_argument('--verbose',action='store_true')
    args=parser.parse_args();result=run(args.data,args.output,args.prior,args.verbose)
    print(json.dumps({'prior':result.prior,'updates':len(result.step_sizes),
          'objective':float(result.objective_history[-1]),'stop_reason':result.stop_reason,
          'runtime_seconds':result.runtime_seconds,'output':args.output}))

if __name__=='__main__':main()
