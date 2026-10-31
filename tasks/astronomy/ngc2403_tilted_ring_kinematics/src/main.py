"""Standard observation-to-kinematics entry point; no hidden assets are read."""
from pathlib import Path
import argparse
import numpy as np
from .solvers import reconstruct


def run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz',seed=-1,progress=None):
    """Actively fit all observations, write three batch-first physical profiles."""
    with np.load(input_path,allow_pickle=False) as archive:
        if 'spectral_cube' not in archive:raise ValueError('Missing spectral_cube')
        observation=archive['spectral_cube']
    if observation.shape!=(1,62,150,150) or observation.dtype!=np.float32:
        raise ValueError('Expected float32 spectral_cube shape(1,62,150,150)')
    field,diagnostics=reconstruct(observation[0],seed=seed,progress=progress)
    destination=Path(output_path);destination.parent.mkdir(parents=True,exist_ok=True)
    result=dict(rotation=field[None,:,0],dispersion=field[None,:,1],position_angle=field[None,:,2])
    np.savez_compressed(destination,**result)
    return result,diagnostics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/raw_data.npz')
    parser.add_argument('--output',default='output/reconstruction.npz')
    parser.add_argument('--seed',type=int,default=-1)
    args=parser.parse_args(argv)
    def progress(stage,ring,field,result):
        print(f'Stage{stage} ring{ring:02d}/41: Vrot={field[0]:.6f} km/s, dispersion={field[1]:.6f} km/s, PA={field[2]:.6f} deg',flush=True)
    run(args.input,args.output,args.seed,progress)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
