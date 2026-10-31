"""Reconstruct all66real time profiles and759height bins."""
import numpy as np
from .preprocessing import ROOT,load_data
from .solvers import reconstruct

def main():
    out,_=reconstruct(load_data());path=ROOT/'output/reconstruction.npz';path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,**out);return path

if __name__=='__main__':main()
