"""Public acquisition headers, preserved verbatim from author .DH files."""
from pathlib import Path
import numpy as np
ACQUISITION=np.array([[5.33304,.00011,.002,201.3,1.9872036],
 [10.00685,.00011,.002,201.3,1.9872036],
 [17.13053,.00011,.002,201.3,1.9872036],
 [24.99623,.00011,.002,201.3,1.9872036],
 [29.99722,.00011,.002,201.3,1.9872036],
 [34.99955,.00011,.002,201.3,1.9872036]],dtype=np.float64)
def load_data(path='data/raw_data.npz'):
    """Return native (experiment, injection) float64 heats and volumes."""
    with np.load(Path(path),allow_pickle=False) as f:
        data={k:f[k][0].copy() for k in ['heats','injection_volumes']}
    for k,a in data.items():
        if a.shape!=(6,14) or a.dtype!=np.float64 or not np.isfinite(a).all():
            raise ValueError('invalid '+k)
    return data
