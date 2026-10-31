"""python -m src.main writes the standard submission."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    s,v=load_data();result=reconstruct(s,v)
    Path('output').mkdir(exist_ok=True)
    np.savez('output/reconstruction.npz',**result)
    return result
if __name__=='__main__':main()
