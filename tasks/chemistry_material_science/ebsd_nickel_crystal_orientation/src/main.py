"""Standard entry point python -m src.main."""
from pathlib import Path
import numpy as np,torch
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    torch.set_num_threads(1);root=Path.cwd();raw,master=load_data(root/'data/raw_data.npz');q=reconstruct(raw,master);target=root/'output/reconstruction.npz';target.parent.mkdir(parents=True,exist_ok=True);np.savez(target,reconstruction=q[None]);print(target)
if __name__=='__main__':main()
