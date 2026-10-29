"""Standard command-line reconstruction: python -m src.main."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .preprocessing import load_data
from .solvers import reconstruct


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw_data.npz')
    parser.add_argument('--output', default='output/reconstruction.npz')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    arguments = parser.parse_args(argv)
    torch.set_num_threads(1)
    observations, mixing = load_data(arguments.data)
    def progress(index, loss, active):
        if index % 25 == 0 or index == 249:
            print(f'iteration={index+1}/250 active={active} loss={loss:.9g}', flush=True)
    result = reconstruct(observations, mixing, device=arguments.device, progress=progress)
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **{key: result[key] for key in
                        ('reconstruction', 'objects', 'wavefront_coefficients')})
    print(f'Runtime: {result["runtime_seconds"]:.3f} s; wrote {destination}', flush=True)
    return result


if __name__ == '__main__':
    main()
