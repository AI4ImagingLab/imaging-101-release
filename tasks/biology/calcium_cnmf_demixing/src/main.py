"""Standard native-data reconstruction entry point."""
from pathlib import Path
import argparse,time
import numpy as np

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    args=parser.parse_args(argv)
    from .preprocessing import load_movie
    from .solvers import reconstruct
    output,review=reconstruct(load_movie(args.data),progress=lambda *x:print(*x,flush=True))
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**output)
    print(f"Accepted {output['reconstruction'].shape[1]} sources; seconds={review['seconds']:.6f}")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
