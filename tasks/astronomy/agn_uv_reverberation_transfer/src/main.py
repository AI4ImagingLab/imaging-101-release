"""Standard workspace-relative executable reconstruction."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def main():
    out,info=reconstruct(load_data());Path('output').mkdir(exist_ok=True);np.savez_compressed('output/reconstruction.npz',**out)
    print('Fresh MAP log objective:',info['logp'],'; selected start:',info['selected_start'])
if __name__=='__main__':main()
