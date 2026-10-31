import time
from pathlib import Path
import numpy as np
from .preprocessing import load
from .solvers import reconstruct

def main():
 t=time.perf_counter();f,r,it=reconstruct(load());Path('output').mkdir(exist_ok=True);np.savez('output/reconstruction.npz',reconstruction=f,remineralization=r);print('seconds',time.perf_counter()-t,'iterations',it.min(),it.max())
if __name__=='__main__':main()
