"""Standard entry point: python -m src.main."""
from pathlib import Path
import argparse
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct,output_arrays


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    args=parser.parse_args(argv)
    model,details=reconstruct(load_observations(args.data))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**output_arrays(model))
    print('Reconstructed all1299fault-slip rates and six nuisance coefficients; weighted RMS={:.12g}'.format(details['weighted_observation_rms']))
    return 0


if __name__=='__main__':raise SystemExit(main())
