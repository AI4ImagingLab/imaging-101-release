"""Bound measured flux and independent calibration; no target-derived inputs."""
from pathlib import Path
import numpy as np

X_PAD=100
Y_PAD=75

def load_data(path):
    with np.load(Path(path),allow_pickle=False) as f:
        return {key:f[key][0].copy() for key in f.files}

def pad_flux(data,mode='linear_ramp'):
    """Preserve the original ramp endpoints and floating-point padded axes."""
    xs,ys=data['xs'],data['ys'];dx=xs[1]-xs[0];dy=ys[1]-ys[0]
    flux=np.pad(data['flux'],((Y_PAD,Y_PAD),(X_PAD,X_PAD)),mode=mode)
    x=np.linspace(xs[0]-X_PAD*dx,xs[-1]+X_PAD*dx,flux.shape[1])
    y=np.linspace(ys[0]-Y_PAD*dy,ys[-1]+Y_PAD*dy,flux.shape[0])
    return flux,x,y
