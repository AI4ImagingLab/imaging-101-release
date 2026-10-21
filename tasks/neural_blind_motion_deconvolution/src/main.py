"""Standard entry point; reconstructs raw data without any hidden assets."""
import argparse
from pathlib import Path
import numpy as np
from .preprocessing import load_observation
from .solvers import reconstruct


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    result=reconstruct(load_observation(args.input),device=args.device)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,reconstruction=result.reconstruction[None],kernel=result.kernel[None])
    print(f'Completed 2500 iterations in {result.runtime_seconds:.3f} s; saved {target}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
