"""Standard independent full51-frame491-point reconstruction."""
from pathlib import Path
import argparse,time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
    """Infer all three declared quantities,save NPZ,return quantities/info/runtime."""
    start=time.perf_counter();shape,cameras,translation,info=reconstruct(load_observations(input_path))
    p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,reconstruction=shape[None],cameras=cameras[None],translations=translation[None])
    return shape,cameras,translation,info,time.perf_counter()-start
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw_data.npz');p.add_argument('--output',default='output/reconstruction.npz');a=p.parse_args()
    *_,seconds=run(a.input,a.output);print(f'Saved {a.output};CPU {seconds:.4f}s')
if __name__=='__main__':main()
