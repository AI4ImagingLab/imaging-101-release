"""Standard CPU reconstruction entry point."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data,reconstruct
def main():
    start=time.perf_counter();result,state=reconstruct(load_data())
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=result[None])
    print('iterations',len(state.history),'chi2/sample',state.history[-1,1]/state.observation.size,'seconds',time.perf_counter()-start)
    return result,state
if __name__=='__main__':main()
