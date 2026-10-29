"""Standard task entry point; imports never mutate plotting backends."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

ROOT=Path(__file__).resolve().parent.parent


def main(input_path=None,output_path=None):
    """Actively reconstruct every bound root and write the sole NPZ output."""
    started=time.perf_counter()
    ratios,low,high=load_observations(ROOT/'data/raw_data.npz' if input_path is None else input_path)
    result,lower,upper,rates=reconstruct(ratios,low,high)
    path=ROOT/'output/reconstruction.npz' if output_path is None else Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,reconstruction=result[None])
    return {'seconds':time.perf_counter()-started,'images':len(result),'lower_ratio':lower,'upper_ratio':upper,'maximum_clipped_fraction':float(rates.max())}


if __name__=='__main__':print(main())
