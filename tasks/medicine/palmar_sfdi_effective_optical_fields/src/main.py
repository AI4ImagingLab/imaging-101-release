from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1];hand,phantom,rcal,freq,cal,observed=load_data(root/'data/raw_data.npz');parameters,predicted=reconstruct(observed,freq);(root/'output').mkdir(exist_ok=True);np.savez(root/'output/reconstruction.npz',reconstruction=parameters.transpose(2,0,1)[None],reflectance=predicted.transpose(2,0,1)[None]);return hand,phantom,rcal,freq,cal,observed,parameters,predicted
if __name__=='__main__':
    result=run();print('Full five-frequency prediction residual (low2 fit)',np.linalg.norm(result[-1]-result[-3])/np.linalg.norm(result[-3]))
