"""Observed reference normalization; no fitted quantities enter support."""
import numpy as np
from scipy.interpolate import interpn

def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as z:
        image=z['measurements'][0];offsets=z['offsets'][0];mask=z['observation_mask'][0]
    if image.shape!=(192,192,62) or offsets.shape!=(62,) or mask.shape!=(192,192):raise ValueError('invalid acquisition shape')
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(offsets)):raise ValueError('nonfinite acquisition')
    return image,offsets,mask

def normalize(image,offsets):
    """Return normalized53-offset spectra, offsets, and drift reference arrays."""
    ref_index=np.where(offsets>15)[0]
    m0=image[:,:,ref_index]
    corrected_offsets=np.delete(offsets,ref_index)
    corrected_images=np.delete(image,ref_index,axis=2)
    points=(np.arange(image.shape[0]),np.arange(image.shape[1]),ref_index)
    kept=np.delete(np.arange(len(offsets)),ref_index)
    xi,yi,fi=np.meshgrid(np.arange(image.shape[0]),np.arange(image.shape[1]),kept,indexing='ij')
    reference=interpn(points,m0,np.stack((xi,yi,fi),axis=-1))
    with np.errstate(divide='ignore',invalid='ignore'):
        spectra=np.nan_to_num(corrected_images/reference)
    return spectra,corrected_offsets,{'m0':m0[:,:,0],'m0_final':m0[:,:,-1],'m0_interp':reference}
