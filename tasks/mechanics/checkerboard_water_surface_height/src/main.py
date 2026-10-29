from pathlib import Path
import time,numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1];start=time.perf_counter();r,d=load_data(root/'data/raw_data.npz');h,_=reconstruct(r,d);(root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',reconstruction=h[None]);print(f'Active reconstruction: {time.perf_counter()-start:.6f} seconds')
if __name__=='__main__':main()
