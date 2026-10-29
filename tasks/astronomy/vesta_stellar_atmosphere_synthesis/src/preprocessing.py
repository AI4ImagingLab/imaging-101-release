"""Published Fe-window observations and masks, without inferred-target calibration."""
from pathlib import Path
import numpy as np

LIGHT_SPEED=299792.458

def load_observations(path):
    """Load NPZ with no pickle and remove the single benchmark batch axis."""
    with np.load(Path(path),allow_pickle=False) as z:return {k:z[k][0].copy() for k in z.files}

def select_ranges(raw):
    """Preserve the original 218-window order and any duplicated wavelength rows."""
    pieces={key:[] for key in ['wavelength','flux','uncertainty']}
    for lo,hi in raw['ranges']:
        center=.5*(lo+hi);width=.5*(hi-lo);selected=np.abs(raw['wavelength']-center)<=width
        for key in pieces:pieces[key].append(raw[key][selected])
    return {key:np.concatenate(values) for key,values in pieces.items()}

def mask_flux(wavelength,flux,uncertainty,masks):
    """Return original wavelength/flux/error and diagnostic masked/ranked velocities.

    Masks contain centre and half-width in angstrom. Overlapping masks duplicate
    their selected observations exactly as the original acquisition convention.
    """
    waves=[];values=[];errors=[]
    for center,width in masks:
        selected=np.abs(wavelength-center)<=width
        waves.append(wavelength[selected]);values.append(flux[selected]);errors.append(uncertainty[selected])
    wave=np.concatenate(waves);value=np.concatenate(values);error=np.concatenate(errors)
    midpoint=np.sqrt(np.max(wave)*np.min(wave))
    velocity=LIGHT_SPEED*(wave-midpoint)/midpoint
    ranked_velocity=LIGHT_SPEED*(wavelength-midpoint)/midpoint
    return wave,value,error,velocity,ranked_velocity
