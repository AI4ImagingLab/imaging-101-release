"""Standard native two-frequency complex shear-modulus reconstruction."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    waves,support,frequencies,coordinates=load_data();begin=time.perf_counter()
    modulus,_,_=reconstruct(waves,frequencies,coordinates,support)
    Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=modulus.transpose(0,3,2,1)[None])
    print(f'Fresh two-frequency AHI: {time.perf_counter()-begin:.6f}s; support={support.sum()}')
    return modulus
if __name__=='__main__':main()
