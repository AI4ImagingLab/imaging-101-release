"""Direct heterodyne demodulation and calibrated spectral inversion."""
from .preprocessing import subtract_background
from .physics_model import calibration,analytic_spectrum,depth_transform
def reconstruct(data,dispersion_correction=True,apodize=True):
    corrected=subtract_background(data['fringes'],data['background'])
    indices,dispersion,window=calibration(data['chirp'],data['dispersion'])
    if not dispersion_correction: dispersion=dispersion*0+1
    if not apodize: window=window*0+1
    spectrum=analytic_spectrum(corrected,indices,dispersion)
    return depth_transform(spectrum,window)[None], {'spectrum':spectrum,'indices':indices,'dispersion':dispersion,'window':window}
