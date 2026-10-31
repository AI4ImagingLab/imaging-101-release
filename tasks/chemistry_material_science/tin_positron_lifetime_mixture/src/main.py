from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
from .physics_model import package

def run(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1];counts,prompt,binned=load_data(root/'data/raw_data.npz');p,pred,history=reconstruct(binned);(root/'output').mkdir(exist_ok=True);np.savez(root/'output/reconstruction.npz',**package(p,pred));return counts,prompt,binned,p,pred,history
if __name__=='__main__':
    start=time.perf_counter();r=run();print('Complete6conditions, seconds',time.perf_counter()-start);print('Lifetimes (ps)',r[3][:,[0,2,4]]*50)
