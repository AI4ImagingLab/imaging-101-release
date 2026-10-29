"""Strict loading of public receiver spectra and legitimate spectral calibration."""
from pathlib import Path
import numpy as np

DELTA_CM=.03
IMAGE_SHAPE=(100,100)
NOISE_STD=5000.
EXPECTED={'projections':(1,40,1200),'magnetic_field_G':(1,1200),'gradients_G_cm':(1,2,40),'reference_spectra':(1,2,1200)}

def load_data(path):
 """Return unbatched float64 arrays; reject incomplete or altered interfaces."""
 with np.load(Path(path),allow_pickle=False) as archive:
  if set(archive.files)!=set(EXPECTED):raise ValueError('Public NPZ must contain exactly the four declared keys')
  values={}
  for key,shape in EXPECTED.items():
   value=archive[key]
   if value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError(f'Invalid shape, float64 dtype or finite values for {key}')
   values[key]=value[0]
 if np.any(np.diff(values['magnetic_field_G'])<=0):raise ValueError('Magnetic-field axis must increase')
 return values

def make_model(data):
 from .physics_model import EPRModel
 return EPRModel(data['magnetic_field_G'],data['gradients_G_cm'],data['reference_spectra'],delta=DELTA_CM,shape=IMAGE_SHAPE)
