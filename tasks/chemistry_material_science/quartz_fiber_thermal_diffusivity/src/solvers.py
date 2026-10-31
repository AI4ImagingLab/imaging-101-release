"""Local closed-form extraction of the selected author fiber inversion."""
import numpy as np
from .preprocessing import native_to_vertical,estimate_angle,orient_and_crop,centerline_profiles,linear_fit,FREQUENCIES
from .physics_model import diffusivity_from_slopes
def reconstruct(amplitude,phase,fit_window=(.2,.6),correct_angle=True):
    """Return submission and diagnostic rows for complete two-frequency camera input.
    Input (2,164,1280), mK/degrees. Optional window/angle switch are ablations.
    """
    rows=[];coeff=[];alpha=[];angle=None
    for i,(a,p) in enumerate(zip(amplitude,phase)):
        av,pv=native_to_vertical(a,p)
        if angle is None:angle=estimate_angle(av) if correct_angle else 0.
        ar,pr,center=orient_and_crop(av,pv,angle)
        distance,profiles=centerline_profiles(ar,pr,center,max_distance=fit_window[1])
        left,right=np.searchsorted(distance,fit_window)
        if right==len(distance):right-=1
        coefficients=np.empty((2,2,2));stderr=np.empty((2,2))
        for side in range(2):
            for mode in range(2):coefficients[side,mode],stderr[side,mode]=linear_fit(distance[left:right],profiles[side,left:right,mode])
        diffusivity=diffusivity_from_slopes(coefficients[:,0,0],coefficients[:,1,0],FREQUENCIES[i])
        rows.append(dict(amplitude=ar,phase=pr,angle=angle,center=center,distance=distance,profiles=profiles,fit_indices=(left,right),coefficients=coefficients,slope_stderr=stderr,diffusivity=diffusivity))
        coeff.append(coefficients);alpha.append(diffusivity)
    coeff=np.array(coeff);products=coeff[:,:,0,0]*coeff[:,:,1,0]
    freq=np.repeat(FREQUENCIES,2);joint_slope=np.dot(freq,products.ravel())/np.dot(freq,freq)
    return {'reconstruction':np.array(alpha)[None],'fit_parameters':coeff[None],'joint_diffusivity':np.array([[abs(np.pi/joint_slope)]])},rows
