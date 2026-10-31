from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct,submission
def main():
    result=reconstruct(load_data());Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',**submission(result))
if __name__=='__main__':main()
