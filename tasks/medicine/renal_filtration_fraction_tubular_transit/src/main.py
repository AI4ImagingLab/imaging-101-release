from pathlib import Path
import time,numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1];start=time.perf_counter()
    result,_=reconstruct(load_data(root/'data/raw_data.npz'))
    (root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',**result)
    print(f'Reconstructed 228 kidneys in {time.perf_counter()-start:.3f} s')
if __name__=='__main__':main()
