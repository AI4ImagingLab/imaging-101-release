"""Standard entry point; the numerical backend is confined to its child process."""
from pathlib import Path
import os
import sys
import subprocess
import time

def main():
    if os.environ.get('QIBA_DCE_BACKEND_WORKER')!='1':
        environment=os.environ.copy()
        environment.update(OPENBLAS_CORETYPE='Prescott',OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1',QIBA_DCE_BACKEND_WORKER='1')
        return subprocess.run([sys.executable,'-m','src.main'],env=environment,
                              shell=False).returncode
    import numpy as np
    from .preprocessing import load_observations
    from .solvers import reconstruct
    root=Path(__file__).resolve().parents[1]
    started=time.perf_counter()
    fields,_=reconstruct(load_observations(root/'data/raw_data.npz'))
    (root/'output').mkdir(exist_ok=True)
    np.savez_compressed(root/'output/reconstruction.npz',**fields)
    print(f'All3000 pixels reconstructed in {time.perf_counter()-started:.6f}s')
    return 0

if __name__=='__main__':raise SystemExit(main())
