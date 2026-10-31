import numpy as np

def load_data(path):
    with np.load(path,allow_pickle=False) as a:raw={k:a[k][0].copy() for k in a.files}
    if raw['measurements'].shape!=(63,7) or raw['membership'].shape!=(63,146):raise ValueError('Invalid bound observation dimensions')
    if not all(np.all(np.isfinite(x)) for x in raw.values()):raise ValueError('Nonfinite observation or calibration')
    corrected=raw['measurements']/raw['control'][:,None]*raw['capacity'][:,None]
    return raw,corrected
