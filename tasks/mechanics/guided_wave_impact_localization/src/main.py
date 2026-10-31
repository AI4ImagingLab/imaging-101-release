import time
from pathlib import Path
import numpy as np
import torch
from .preprocessing import load
from .solvers import reconstruct

def main():
 torch.set_num_threads(1);t=time.perf_counter();result,details=reconstruct(load());Path('output').mkdir(exist_ok=True);np.savez('output/reconstruction.npz',**result);print('seconds',time.perf_counter()-t,'final training loss',details['loss_history'][-1])
if __name__=='__main__':main()
