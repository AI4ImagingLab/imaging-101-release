from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    maps,prediction,covariance=reconstruct(load_data());Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=maps[0][None],synchrotron=maps[1][None],dust=maps[2][None])
    print('Wrote three spatial component maps:',maps.shape)
if __name__=='__main__':main()
