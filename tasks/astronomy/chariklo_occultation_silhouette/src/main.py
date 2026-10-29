"""Run the complete five-positive/one-negative chord inverse problem."""
import numpy as np
from .preprocessing import ROOT,load_data
from .solvers import reconstruct

def main():
    result,_=reconstruct(load_data());path=ROOT/'output/reconstruction.npz';path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,**result);return path

if __name__=='__main__':main()
