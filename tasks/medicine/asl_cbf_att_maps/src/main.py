"""Standard task entry point; importing does not select a plotting backend."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_observations
from .solvers import reconstruct

def main():
    """Reconstruct public data and write output/reconstruction.npz; return maps."""
    started=time.perf_counter();result=reconstruct(**load_observations())
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',**result)
    print(f'Wrote output/reconstruction.npz in {time.perf_counter()-started:.3f} s')
    return result

if __name__=='__main__':main()
