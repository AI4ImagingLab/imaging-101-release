"""Public calibrated angular scan and detector intensity."""
from pathlib import Path
import numpy as np
ANGULAR_INCREMENT=5.
RETARDANCE=np.pi/2
PSG_POLARIZER_ANGLE=0.
PSA_POLARIZER_ANGLE=0.

def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:
        if z['angles'].shape!=(1,24) or z['power'].shape!=(1,24):raise ValueError('Expected complete24-angle native scan')
        angles=z['angles'][0].copy();power=z['power'][0].copy()
    if not np.isfinite(angles).all() or not np.isfinite(power).all():raise ValueError('Nonfinite acquisition')
    return angles,power
