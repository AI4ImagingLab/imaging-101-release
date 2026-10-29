"""Full two-standard camera-to-Brillouin-shift reconstruction.
Adapted numerical equations from bmlab developers, GPL-3.0; see LICENSE.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from .preprocessing import find_points,interpolate_time,extract_spectrum
from .physics_model import arc_coordinates,setup,fit_lorentz,fit_double_lorentz,fit_vipa,VIPA,spectral_model
REGIONS=np.array([[4e9,6e9],[9e9,11e9],[-2e9,2e9],[13e9,17e9]])

def calibration_regions(spectra):
    """Identify the two elastic peaks and two inelastic pairs from calibrators."""
    y=np.mean(spectra,axis=0);base=np.nanmedian(y)
    peaks,props=find_peaks(y,prominence=15,width=True,height=15+base)
    if len(peaks)<6:peaks,props=find_peaks(y,prominence=15,width=True)
    if len(peaks)==6:center=np.mean(peaks[2:4])
    else:
        y[y<base]=0;center=np.nansum(y*np.arange(1,len(y)+1))/np.nansum(y)
        if (peaks>center).sum()<3:center=np.mean(peaks[-4:-2])
        elif (peaks<=center).sum()<3:center=np.mean(peaks[2:4])
    n=(peaks<=center).sum()
    def region(i):
        q=(peaks[i]+props['widths'][i]*np.array([-4,4])).astype(int)
        q[q>len(y)]=len(y)
        return q
    r=[region(i) for i in [n-3,n+2,n-2,n-1,n,n+1]]
    return np.array([r[0],r[1],[r[2][0],r[3][1]],[r[4][0],r[5][1]]])

def fit_region(region,x,y,double=False,power=4):
    """Nearest-index, right-exclusive ROI and preserved quartic spectral fit."""
    left=np.nanargmin(abs(x-region[0]));right=np.nanargmin(abs(x-region[1]))
    x=x[left:right];y=y[left:right];mask=np.isfinite(x)&np.isfinite(y);x=x[mask];y=y[mask]
    if double:
        centers,widths,heights,offset=fit_double_lorentz(x,y)
        return np.array(centers),np.array(widths),np.array(heights),offset
    if power==4:return np.array(fit_lorentz(x,y))
    p=np.array(fit_lorentz(x,y));span=x[-1]-x[0]
    result=least_squares(lambda z:y-spectral_model(x,z),p,bounds=([x[0],span/len(x),0,0],[x[-1],span,2*y.max(),2*y.max()]))
    return result.x

def reconstruct(raw,arc_width=2,calibration_mode='interpolate',power=4):
    """Return all declared arrays and independently inspectable intermediate stages.

    Defaults reproduce the complete author Water.h5 pipeline. Optional width,
    first-calibration and L2 variants are labeled notebook sensitivity experiments.
    """
    images=raw['images'][0];cal=raw['calibration_images'][0];ct=raw['calibration_time'][0]
    mt=raw['measurement_time'][0];exposure=raw['exposure'][0]
    points=[find_points(v) for v in cal]
    geometry=[arc_coordinates(v,width=arc_width) for v in points]
    arcs=np.array([v[0] for v in geometry]);circles=np.array([v[1] for v in geometry])
    calibration_spectra=np.array([[extract_spectrum(img,arc) for img in frames] for frames,arc in zip(cal,arcs)])
    regions=np.array([calibration_regions(v) for v in calibration_spectra]);pixel=np.arange(500)
    instrument=setup();calibration_parameters=[];peak_positions=[]
    for spectra,rs in zip(calibration_spectra,regions):
        params=[];positions=[]
        for spectrum in spectra:
            peaks=[fit_region(r,pixel,spectrum)[0] for r in rs[:2]]
            for r in rs[2:]:peaks.extend(fit_region(r,pixel,spectrum,double=True)[0])
            peaks=np.sort(peaks);positions.append(peaks);params.append(fit_vipa(peaks,instrument))
        calibration_parameters.append(params);peak_positions.append(positions)
    calibration_parameters=np.asarray(calibration_parameters)
    calibration_frequency=np.array([[VIPA(pixel,p)-instrument.f0 for p in ps] for ps in calibration_parameters])
    times=mt[:,None]+exposure[:,None]*np.arange(images.shape[1])[None,:]
    measurement_arcs=interpolate_time(ct,arcs,mt)
    spectra=np.array([[extract_spectrum(img,arc) for img in frames] for frames,arc in zip(images,measurement_arcs)])
    means=np.mean(calibration_frequency,axis=1)
    frequencies=interpolate_time(ct,means,times) if calibration_mode=='interpolate' else np.broadcast_to(means[0],(*times.shape,500)).copy()
    parameters=np.array([[[fit_region(r,x,y,power=power) for r in REGIONS] for x,y in zip(xs,ys)] for xs,ys in zip(frequencies,spectra)])
    shifts=np.min(abs(parameters[:,:,:2,0,None]-parameters[:,:,None,2:,0]),axis=-1)
    outputs=dict(reconstruction=shifts[None],peak_parameters=parameters[None],frequency_axis=frequencies[None],extracted_spectra=spectra[None],calibration_frequency=calibration_frequency[None],calibration_arcs=arcs[None],calibration_parameters=calibration_parameters[None])
    details=dict(points=np.array(points),circles=circles,calibration_spectra=calibration_spectra,calibration_regions=regions,calibration_peak_positions=np.array(peak_positions),measurement_arcs=measurement_arcs)
    return outputs,details
