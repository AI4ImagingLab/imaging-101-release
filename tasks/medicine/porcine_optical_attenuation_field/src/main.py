"""Standard full-observation inverse entry point: python -m src.main."""
from pathlib import Path
import time,json,numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
from .physics_model import consistency
def run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz'):
 start=time.perf_counter();signal=load_data(data_path);mu,info=reconstruct(signal)
 output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,reconstruction=mu[None].astype(np.float64))
 return mu,info,time.perf_counter()-start
def main():
 mu,info,seconds=run();print(json.dumps(dict(seconds=seconds,shape=[1,*mu.shape],mask_pixels=int(info['mask'].sum()),forward_residual=consistency(mu,info['term']))))
if __name__=='__main__':main()
