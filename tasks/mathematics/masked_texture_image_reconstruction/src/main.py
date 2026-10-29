"""Standard full-resolution entry point, with no plotting side effects."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def main():
    start=time.perf_counter();y,mask=load_observations();result=reconstruct(y,mask)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=result['u'][None])
    print(f'Full native TGV2 reconstruction completed in {time.perf_counter()-start:.6f}s')
    return result
if __name__=='__main__':main()

