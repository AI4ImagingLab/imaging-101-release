"""Explicit author-selected full spectral preprocessing and native precision."""
import numpy as np

REST_WAVELENGTH=6562.8
SYSTEMIC_KM_S=36.
WAVELENGTH_HALF_WINDOW=35.
SOURCE_TWO_PI=float(8*np.arctan(np.float32(1.)))
PHASE_WIDTH=.03930

def prepare(raw):
    """Prepare all28 observed spectra; no hidden calibration or inverse arrays."""
    wavelength=np.asarray(raw['wavelength']);spectrum=np.asarray(raw['spectrum']);phase=np.asarray(raw['orbital_phase'])
    if wavelength.shape!=(1,28,2048) or spectrum.shape!=wavelength.shape or phase.shape!=(1,28):raise ValueError('Unexpected native observations')
    if any(a.dtype.kind!='f' or not np.isfinite(a).all() for a in [wavelength,spectrum,phase]):raise ValueError('Expected finite floating observations')
    if np.any(np.diff(wavelength,axis=-1)<=0):raise ValueError('Wavelength must increase')
    wave=wavelength[0,0];flux=np.stack([spectrum[0,0]]+[np.interp(wave,wavelength[0,i],spectrum[0,i]) for i in range(1,28)])
    beta=SYSTEMIC_KM_S/299800.
    source_factor=(1+beta)*np.sqrt(1-beta**2)
    corrected=wave/source_factor
    line=(corrected>REST_WAVELENGTH-WAVELENGTH_HALF_WINDOW)&(corrected<REST_WAVELENGTH+WAVELENGTH_HALF_WINDOW)
    band=((corrected>6500)&(corrected<6537))|((corrected>6591)&(corrected<6620))
    selected_wave=corrected[line]
    ratio=(selected_wave/REST_WAVELENGTH)**2
    original_velocity=(ratio-1)*299700./(1+ratio)
    velocity=np.linspace(original_velocity[0],original_velocity[-1],len(original_velocity))
    result=[]
    for row in flux:
        coefficients=np.polyfit(corrected[band],row[band],3)
        continuum=np.polyval(coefficients,selected_wave)
        result.append(np.interp(velocity,original_velocity,row[line]-continuum))
    unquantized=np.array(result)
    # This is the original file-interface precision, explicitly part of the binding.
    spectra=np.array([[float(f'{v:.5f}') for v in row] for row in unquantized])
    vquant=np.array([float(f'{v*1e5:.5e}') for v in velocity])
    pquant=np.array([float(f'{float(np.float32(v)):.6f}') for v in phase[0]])
    return dict(line_spectra=spectra,unquantized_line_spectra=unquantized,velocity_cm_s=vquant,phase_radians=pquant*SOURCE_TWO_PI,phase_width_radians=np.full(28,PHASE_WIDTH*SOURCE_TWO_PI),normalised_wave=selected_wave)
