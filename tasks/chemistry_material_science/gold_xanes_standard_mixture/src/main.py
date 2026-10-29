"""CLI: reconstruct every original comparison, write selected prediction."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data,prepare
from .solvers import solve
def reconstruct(path):
    energy,observation,standards=prepare(load_data(path))
    result=solve(observation,standards)
    return energy,observation,standards,result
def main():
    root=Path(__file__).resolve().parents[1]
    *_,r=reconstruct(root/'data/raw_data.npz');i=r['best']
    (root/'output').mkdir(exist_ok=True)
    np.savez(root/'output/reconstruction.npz',weights=r['weights'][i:i+1],reconstruction=r['fits'][i:i+1])
    print('Selected model',i+1,'weights',r['weights'][i])
if __name__=='__main__':main()
