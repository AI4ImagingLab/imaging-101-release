from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .initialization import initial_energy
from .solvers import reconstruct

def run(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1];raw,corrected=load_data(root/'data/raw_data.npz');initial=initial_energy(raw);energy,uptake,history,obj=reconstruct(raw,corrected,initial);(root/'output').mkdir(exist_ok=True);np.savez(root/'output/reconstruction.npz',reconstruction=energy[None],uptake=uptake[None]);return raw,corrected,initial,energy,uptake,history,obj
if __name__=='__main__':
    start=time.perf_counter();r=run();print('Objective',r[-1],'iterations',len(r[-2]),'seconds',time.perf_counter()-start)
