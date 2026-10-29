"""Run complete native count-image reconstruction without external domain code."""
import numpy as np
from .preprocessing import ROOT,load_data
from .solvers import reconstruct

def main():
    result,diagnostics=reconstruct(load_data())
    path=ROOT/'output/reconstruction.npz';path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,**result)
    print('Reconstructed hydrogen density and emissivity for35spherical shells')
    return path

if __name__=='__main__':main()
