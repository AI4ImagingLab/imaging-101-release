"""Run from task workspace: python -m src.main."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data,amplitude_stack
from .solvers import reconstruct
def main():
    start=time.perf_counter();root=Path(__file__).resolve().parents[1]
    a=amplitude_stack(load_data(root/'data/raw_data.npz'));u,h=reconstruct(a)
    out=root/'output/reconstruction.npz';out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out,reconstruction=u[None].astype('complex128'))
    print(f'Wrote {out}; elapsed {time.perf_counter()-start:.3f} s; amplitude residual {h[-1,1]:.8g}')
if __name__=='__main__':main()
