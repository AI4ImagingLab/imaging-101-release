"""Bound author-standard full-field multispectral observations and independent spectra."""
from pathlib import Path
import numpy as np
FIELD_OF_VIEW=.024975
def load_data(path="data/raw_data.npz"):
    """Return images(T,W,Z,Y,X)float32,spectra(C,W)float64,labels(Z,Y,X)uint8,wavelength(W)nm,times(T,W)s."""
    with np.load(Path(path),allow_pickle=False) as z:data={k:z[k] for k in z.files}
    shapes={"images":(1,76,10,1,111,111),"spectra":(1,2,10),"wavelength":(1,10),"roi_labels":(1,1,111,111),"acquisition_time":(1,76,10)}
    if set(data)!=set(shapes):raise ValueError("Invalid public keys")
    for k,shape in shapes.items():
        dtype=np.float32 if k=="images" else np.uint8 if k=="roi_labels" else np.float64
        if data[k].shape!=shape or data[k].dtype!=dtype or not np.isfinite(data[k]).all():raise ValueError("Invalid "+k)
    if not np.isin(data['roi_labels'],[0,1,2,3]).all():raise ValueError('Invalid ROI labels')
    if not np.all(np.diff(data['wavelength'][0])>0) or np.any(data['spectra']<=0):raise ValueError('Invalid wavelength or spectra')
    if not np.all(np.diff(data['acquisition_time'][0].ravel())>0):raise ValueError('Times must increase')
    return data['images'][0],data['spectra'][0],data['roi_labels'][0],data['wavelength'][0],data['acquisition_time'][0]
