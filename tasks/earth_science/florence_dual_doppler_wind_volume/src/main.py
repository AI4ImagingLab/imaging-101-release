from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    winds,_,state=reconstruct(load_data());Path('output').mkdir(exist_ok=True)
    np.savez_compressed('output/reconstruction.npz',reconstruction=winds[0][None],northward=winds[1][None],vertical=winds[2][None]);print(state)
if __name__=='__main__':main()
