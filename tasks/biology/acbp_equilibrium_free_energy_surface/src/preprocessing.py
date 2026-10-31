"""Complete observation-derived normalization; no fitted calibration."""
import numpy as np
NORMALIZATION_CELSIUS=70.
def load_observations(path):
 with np.load(path,allow_pickle=False) as z:return z['temperature_celsius'][0].copy(),z['denaturant_molar'][0].copy(),z['intensity'][0].copy()
def normalize_intensity(temperature,intensity):
 indices=np.argmin(abs(temperature-NORMALIZATION_CELSIUS),axis=1)
 factors=intensity[:,np.arange(len(temperature)),indices]
 if np.any(factors==0):raise ValueError('Zero normalization observation')
 return intensity/factors[:,:,None]
def flatten_observations(temperature,denaturant,intensity):
 t=np.tile(temperature.ravel(),2);d=np.tile(np.repeat(denaturant,temperature.shape[1]),2);c=np.repeat(np.arange(2),temperature.size)
 return t,d,c,intensity.ravel()
