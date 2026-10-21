from pathlib import Path
import numpy as np,torch
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    torch.set_num_threads(1);data=load_data();result=reconstruct(data);Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',**result)
if __name__=='__main__':main()
