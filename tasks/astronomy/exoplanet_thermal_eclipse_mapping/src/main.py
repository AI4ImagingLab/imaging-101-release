from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    image,state=reconstruct(load_data());Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',reconstruction=image[None]);print('Minimum ESS',state['effective_sample_size'].min(),'maximum tau',state['autocorrelation'].max())
if __name__=='__main__':main()
