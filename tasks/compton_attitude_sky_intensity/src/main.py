"""Standard command: python -m src.main."""
from pathlib import Path
import argparse,json

def main(argv=None):
    from .preprocessing import load_observations
    from .solvers import reconstruct
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw_data.npz')
    parser.add_argument('--output',type=Path,default=Path('output/reconstruction.npz'))
    parser.add_argument('--iterations',type=int,default=30)
    args=parser.parse_args(argv)
    data=load_observations(args.input)
    def progress(iteration,model,delta,likelihood):
        print(json.dumps(dict(iteration=iteration,log_likelihood=likelihood)),flush=True)
    result=reconstruct(data,iterations=args.iterations,callback=progress)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,reconstruction=result.reconstruction)
    print(json.dumps(dict(output=str(args.output),iterations=len(result.maps),seconds=result.seconds)),flush=True)
    return 0

if __name__=='__main__':
    import os
    for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[variable]='1'
    raise SystemExit(main())
