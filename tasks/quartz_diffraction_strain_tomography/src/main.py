"""Standard full six-component reconstruction entry point."""
from pathlib import Path
import time,numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    root=Path(__file__).resolve().parents[1];data=load_data(root/'data/raw_data.npz');start=time.perf_counter();field,diagnostics=reconstruct(data)
    (root/'output').mkdir(exist_ok=True);np.savez_compressed(root/'output/reconstruction.npz',reconstruction=field[None]);print('Active seconds:',time.perf_counter()-start,'iterations:',diagnostics['iterations'],'objective:',diagnostics['objective'])
if __name__=='__main__':main()
