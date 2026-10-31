import time
from pathlib import Path
import numpy as np
from .preprocessing import load
from .solvers import reconstruct

def main():
    t=time.perf_counter(); image,background,data=load();result,diagnostics=reconstruct(data)
    Path('output').mkdir(exist_ok=True);np.savez('output/reconstruction.npz',reconstruction=result)
    print(result,diagnostics,'seconds',time.perf_counter()-t)
if __name__=='__main__':main()
