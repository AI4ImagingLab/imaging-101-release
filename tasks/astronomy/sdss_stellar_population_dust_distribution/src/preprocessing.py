"""Public calibrated observations and independent SSP templates."""
import numpy as np
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as z:return {k:z[k][0] for k in z.files}
def normalize_models(data,models):
    return models/np.median(models,axis=1)[:,None]*np.median(data)
