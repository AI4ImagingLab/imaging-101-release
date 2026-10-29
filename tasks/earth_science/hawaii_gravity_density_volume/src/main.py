"""Standard task entry point; imports never select a plot backend."""
from pathlib import Path
import time,numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    start=time.perf_counter();density,state=reconstruct(load_data())
    Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',reconstruction=density[None])
    print('seconds',time.perf_counter()-start,'RMS mGal',np.sqrt(np.mean((state['prediction']-state['anomaly'])**2)))
    return density,state
if __name__=='__main__':main()
