"""Standard entry point: python -m src.main."""
import argparse
from pathlib import Path
import time
import numpy as np

from .preprocessing import load_data
from .solvers import reconstruct


def save_submission(output, path='output/reconstruction.npz'):
    """Save all five batch-first float32 quantities to the single contract file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{key: value[None].astype(np.float32) for key, value in output.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/raw_data.npz')
    parser.add_argument('--output', default='output/reconstruction.npz')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    output, diagnostics = reconstruct(load_data(args.data), workers=args.workers)
    save_submission(output, args.output)
    print(f'Wrote {args.output}; {len(diagnostics["regions"])} common regions; {time.perf_counter()-start:.3f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
