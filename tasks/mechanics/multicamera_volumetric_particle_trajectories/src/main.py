"""Standard entry point: python -m src.main."""
from pathlib import Path
import argparse,json
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data/raw_data.npz'));parser.add_argument('--output',type=Path,default=Path('output/reconstruction.npz'));parser.add_argument('--diagnostics',type=Path,default=None);args=parser.parse_args(argv)
    result,diagnostics=reconstruct(load_data(args.data));args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**result)
    if args.diagnostics is not None:
        args.diagnostics.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.diagnostics,**diagnostics)
    print(json.dumps({'status':'completed','rows':result['positions'].shape[1],'runtime_seconds':diagnostics['runtime_seconds']},allow_nan=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
