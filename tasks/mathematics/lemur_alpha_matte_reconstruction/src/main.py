"""Standard standalone reconstruction entry point."""
from pathlib import Path
import argparse,time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct


def run(input_path='data/raw_data.npz', output_path='output/reconstruction.npz'):
    image, trimap = load_data(input_path)
    start = time.perf_counter()
    alpha, diagnostics = reconstruct(image, trimap)
    p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, reconstruction=alpha[None])
    print(f'Fresh opacity: {time.perf_counter()-start:.6f}s; CG iterations {len(diagnostics["history"])}; relative residual {diagnostics["explicit_relative_residual"]:.12g}')
    return alpha, diagnostics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--input', default='data/raw_data.npz'); parser.add_argument('--output', default='output/reconstruction.npz')
    args = parser.parse_args(); run(args.input, args.output)
