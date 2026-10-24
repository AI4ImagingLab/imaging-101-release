"""Standard CLI: python -m src.main."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct
def main():
    root=Path.cwd();counts,irf=load_data(root/'data/raw_data.npz');depth,_=reconstruct(counts,irf);out=root/'output/reconstruction.npz';out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,reconstruction=depth[None]);print(out)
if __name__=='__main__':main()
