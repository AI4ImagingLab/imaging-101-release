from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    output,_=reconstruct(load_data());Path('output').mkdir(exist_ok=True);np.savez('output/reconstruction.npz',**output)
if __name__=='__main__':main()
