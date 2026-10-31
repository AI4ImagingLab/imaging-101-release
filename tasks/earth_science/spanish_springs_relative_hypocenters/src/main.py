import time
from pathlib import Path
import numpy as np
from .preprocessing import load
from .solvers import reconstruct

def main():
 t=time.perf_counter();result,_=reconstruct(load(),verbose=True);Path('output').mkdir(exist_ok=True);np.savez('output/reconstruction.npz',**result);print('complete seconds',time.perf_counter()-t)
if __name__=='__main__':main()
