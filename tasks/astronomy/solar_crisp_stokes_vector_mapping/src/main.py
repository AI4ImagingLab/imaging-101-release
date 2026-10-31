"""Standard active reconstruction; does not change a plotting backend on import."""
import numpy as np
from .preprocessing import ROOT,load_data
from .solvers import reconstruct

def main():
    parameters,diagnostics=reconstruct(load_data())
    output=ROOT/'output/reconstruction.npz'; output.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(output,reconstruction=parameters[None])
    print('Saved reconstruction.npz; mean weighted chi-square',float(diagnostics['chi2_cycle2'].mean()))
    return output
if __name__=='__main__': main()
