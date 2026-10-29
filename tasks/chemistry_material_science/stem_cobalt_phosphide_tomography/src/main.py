from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1];start=time.perf_counter()
    d,a=load_data(root/'data/raw_data.npz');v,h=reconstruct(d,a)
    (root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',reconstruction=v[None])
    print(f'Wrote output/reconstruction.npz; elapsed {time.perf_counter()-start:.3f}s; residual {h[-1,1]:.9g}')
if __name__=='__main__':main()
