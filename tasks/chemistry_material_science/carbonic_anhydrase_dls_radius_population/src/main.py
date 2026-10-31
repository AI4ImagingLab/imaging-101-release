from pathlib import Path
import numpy as np
from .preprocessing import load_data,transform
from .physics_model import relaxation_times
from .solvers import solve

def run(root=None):
    root=Path(root) if root else Path(__file__).resolve().parents[1];t,g,r,lag_us=load_data(root/'data/raw_data.npz');beta,field,valid=transform(t,g,lag_us);s=relaxation_times(r);p,pred,res,pen=solve(t,g,s,beta,field,valid);(root/'output').mkdir(exist_ok=True);np.savez(root/'output/reconstruction.npz',reconstruction=p[None],correlation=pred.T[None]);return t,g,r,s,beta,field,valid,p,pred,res,pen
if __name__=='__main__':
    result=run();print('Reconstructed two spectra',result[7].shape)
