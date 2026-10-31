import time
from pathlib import Path
import numpy as np
from .preprocessing import load
from .solvers import reconstruct

def main():
 t=time.perf_counter();result,info=reconstruct(load());Path('output').mkdir(exist_ok=True);np.savez('output/reconstruction.npz',**result);print('seconds',time.perf_counter()-t,'mean selected pre-update MAE',info['selection_mae'].mean())
if __name__=='__main__':main()
