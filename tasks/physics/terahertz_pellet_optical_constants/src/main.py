from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    t,e=load_data();result=reconstruct(t,e);Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',**result)
if __name__=='__main__':main()
