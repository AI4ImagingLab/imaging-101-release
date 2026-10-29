from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    image,_,result=reconstruct(load_data());Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=image[None])
    print('Spatial current fractions:',image.shape,'sum',image.sum(),'optimality',result.optimality)
if __name__=='__main__':main()
