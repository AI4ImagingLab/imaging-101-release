import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0].copy() for k in f.files}
def normalize_observations(data):
    """Observed radial RMS over the declared binary radar coverage only."""
    w=data['validity'];return float(np.sqrt(np.sum(data['radial_velocity']**2*w)/np.sum(w)))
