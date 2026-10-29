from pathlib import Path
import numpy as np
N_LATITUDE=180
N_LONGITUDE=360
RADIAL_CELLS=35
SOURCE_RADIUS=2.5
LONGITUDE_EDGE=22.0
def load_data(path='data/raw_data.npz'):
    """Native float32 radial magnetic map (180,360), Gauss; south to north."""
    with np.load(Path(path),allow_pickle=False)as f:
        x=f['magnetogram']
        if x.shape!=(1,180,360)or x.dtype.kind!='f' or x.dtype.itemsize!=4 or not np.isfinite(x).all():raise ValueError('Expected finite float32 magnetogram (1,180,360)')
        return x[0].copy()
def balance_flux(magnetogram):
    """Equal-area measured-map mean removal, float64; no target calibration."""
    x=np.asarray(magnetogram,dtype=np.float64)
    return x-np.mean(x,dtype=np.float64)
