"""Standard CPU entry point for joint variational motion deblatting."""
import argparse
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data, OBJECT_SHAPE
from .solvers import Params, estimateFMH


def reconstruct(observation, background, projection_mode='exact'):
    """Return appearance, opacity and exposure kernel from one measured crop."""
    parameters = Params()
    if projection_mode not in ('exact', 'upstream_cyclic'):
        raise ValueError('Unknown projection mode.')
    parameters.projection_mode = projection_mode
    parameters.verbose = False
    parameters.visualize = False
    kernel, appearance, mask = estimateFMH(observation, background,
                                         np.ones(OBJECT_SHAPE), params=parameters)
    return {'reconstruction': appearance[None], 'mask': mask[None], 'kernel': kernel[None]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw_data.npz')
    parser.add_argument('--output', default='output/reconstruction.npz')
    parser.add_argument('--projection', choices=('exact','upstream_cyclic'), default='exact')
    args = parser.parse_args()
    start = time.perf_counter()
    result = reconstruct(*load_data(args.data), projection_mode=args.projection)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f'Reconstructed all three unknowns in {time.perf_counter()-start:.3f} s.')


if __name__ == '__main__':
    main()
