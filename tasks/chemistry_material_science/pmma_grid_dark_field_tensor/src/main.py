from pathlib import Path
import time,numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    root=Path(__file__).resolve().parents[1];start=time.perf_counter();grid,sample=load_data(root/'data/raw_data.npz');c,t,_=reconstruct(grid,sample);(root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',reconstruction=c[None],transmission=t[None]);print(f'Active reconstruction seconds: {time.perf_counter()-start:.6f}')
if __name__=='__main__':main()
