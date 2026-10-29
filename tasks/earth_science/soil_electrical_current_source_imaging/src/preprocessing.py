import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0].copy() for k in f.files}
def observation_weights(observations,error_floor=1.):
    """Original inverse sqrt(abs transfer resistance), capped at error_floor."""
    with np.errstate(divide='ignore'):w=1/np.sqrt(np.abs(observations))
    w[w>=error_floor]=error_floor
    return w
