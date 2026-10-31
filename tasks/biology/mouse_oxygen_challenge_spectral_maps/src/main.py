"""Standard full76-cycle two-chromophore and apparent oxygenation inverse."""
from pathlib import Path
import time
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    images,spectra,labels,wavelength,times=load_data();begin=time.perf_counter()
    coefficients,saturation=reconstruct(images,spectra,labels)
    Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',reconstruction=coefficients[None],saturation=saturation[None])
    print(f'Fresh full76-cycle inverse {time.perf_counter()-begin:.6f}s, fixed pixels={(labels>0).sum()}')
    return coefficients,saturation
if __name__=='__main__':main()
