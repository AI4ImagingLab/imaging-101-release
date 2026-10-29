"""Standard standalone CPU execution; import has no plotting/backend side effects."""
from pathlib import Path
import time
import numpy as np
import torch
from .preprocessing import load_observations
from .solvers import reconstruct

def main() -> None:
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    start=time.perf_counter()
    result=reconstruct(load_observations(root/'data/raw_data.npz'))
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',reconstruction=result['reconstruction'][None])
    print(f'Full CPU reconstruction: {time.perf_counter()-start:.6f}s')

if __name__=='__main__':main()
