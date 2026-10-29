"""Standard full-volume reconstruction entry point; never reads hidden assets."""
import argparse
from pathlib import Path
import json
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct


def main(argv=None):
    """Read public data, reconstruct, and write only the declared NPZ result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args(argv)
    observations = load_data()
    reconstruction, diagnostics = reconstruct(observations['phase'], observations['mask'], device=args.device)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz', reconstruction=reconstruction[None],
                        residual_phase=diagnostics['residual_phase'][None],
                        auxiliary_w=diagnostics['auxiliary_w'][None])
    print(json.dumps({k: v for k, v in diagnostics.items() if k in ('iterations', 'runtime_seconds')}, allow_nan=False))
    return diagnostics


if __name__ == '__main__':
    main()
