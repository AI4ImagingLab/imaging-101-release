"""Standard independent entry point: python -m src.main."""
from pathlib import Path
import numpy as np
import torch
from .preprocessing import load_pressure
from .solvers import solve
def main():
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1];p=load_pressure(root/'data/raw_data.npz')
    q,_=solve(p)
    output=root/'output/reconstruction.npz';output.parent.mkdir(exist_ok=True)
    np.savez_compressed(output,reconstruction=q[None]);return output
if __name__=='__main__':print(main())
