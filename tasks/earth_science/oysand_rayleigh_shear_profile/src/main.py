"""Complete original CPSO inversion of all real dispersion points."""
import numpy as np
from .preprocessing import ROOT,load_data
from .solvers import reconstruct

def main():
    out,_=reconstruct(load_data());path=ROOT/'output/reconstruction.npz';path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,**out);return path

if __name__=='__main__':main()
