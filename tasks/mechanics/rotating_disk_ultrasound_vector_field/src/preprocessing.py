"""Measured RF and known grid/support only; no estimated velocities as inputs."""
from pathlib import Path
import numpy as np

def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as d:
        if set(d.files)!={'rf','observation_roi','x','z'}:raise ValueError('Unexpected observation keys')
        rf=d['rf'];roi=d['observation_roi'];x=d['x'];z=d['z']
        if rf.shape!=(1,32,128,334) or rf.dtype!=np.int16:raise ValueError('Expected measured int16 RF batch/ensemble/element/time')
        if roi.shape!=(1,251,251) or roi.dtype!=np.bool_:raise ValueError('Invalid public disk support')
        if x.shape!=(1,251) or z.shape!=(1,251) or not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('Invalid known coordinate axes')
        radius=np.hypot(x[0][None,:],z[0][:,None]-.023)
        expected=radius<.01
        # Preserve the supplied author's exact mask at floating-point circle
        # boundary pixels; do not silently choose a different hypot rounding.
        away_from_boundary=abs(radius-.01)>64*np.finfo(float).eps*.01
        if not np.array_equal(roi[0][away_from_boundary],expected[away_from_boundary]):raise ValueError('Disk support disagrees with known geometry')
        return {'rf':rf[0].transpose(2,1,0).astype(float),'roi':roi[0].copy(),'x':x[0].copy(),'z':z[0].copy()}
