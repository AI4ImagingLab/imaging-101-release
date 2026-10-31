from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    maps,_=reconstruct(load_data());Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=maps[0][None],surface_density=maps[1][None])
    print('Wrote colour temperature K and apparent surface density kg/m^2',maps.shape)
if __name__=='__main__':main()
