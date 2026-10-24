"""Standard CPU reconstruction entry point."""
from pathlib import Path
import time,numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct
def main():
    t=time.perf_counter();image,_=reconstruct(*load_observations());Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=image[None].astype(np.float64))
    print(f'Reconstructed relative log radiance in {time.perf_counter()-t:.6f} seconds')
if __name__=='__main__':main()
