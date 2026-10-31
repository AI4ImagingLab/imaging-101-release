from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import solve
def reconstruct(path):
    image,contour=load_data(path)
    return image,contour,solve(contour)
def main():
    root=Path(__file__).resolve().parents[1];*_,r=reconstruct(root/'data/raw_data.npz')
    (root/'output').mkdir(exist_ok=True)
    np.savez(root/'output/reconstruction.npz',**{k:r[k][None] for k in ('parameters','reconstruction','capillary_length')})
    print(r['parameters'])
if __name__=='__main__':main()
