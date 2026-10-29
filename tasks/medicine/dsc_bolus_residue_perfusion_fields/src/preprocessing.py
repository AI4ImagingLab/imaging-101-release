"""GRE signal-to-relaxivity transformation, adapted from MIT DSC toolbox.

Copyright original authors, University of Padova; see ../LICENSE.
The documented slice-stride correction is implemented explicitly.
"""
import numpy as np

TE=0.025
TR=1.55

def signal_to_concentration(signal,support,te=TE):
    """Return concentration[slice,row,col,time], baseline and22 baseline counts.

    Input intensities are ADU; only the acquisition-time input support is used.
    The concentration proxy is s^-1; no relaxivity calibration is invented.
    """
    signal=np.asarray(signal);support=np.asarray(support)
    if signal.ndim!=4 or support.shape!=signal.shape[:-1] or support.dtype!=bool:
        raise ValueError('Expected slice,row,column,time signal and boolean spatial support')
    if not np.isfinite(signal).all() or np.any(signal[support]<=0) or te<=0:
        raise ValueError('Finite data and strictly positive supported intensities required')
    concentration=np.zeros(signal.shape,dtype=np.float64)
    baseline=np.zeros(support.shape,dtype=np.float64);bolus=np.zeros(len(signal),dtype=np.int64)
    for z in range(len(signal)):
        selected=np.asarray(signal[z][support[z]],dtype=np.float64)
        # Source empty-slice mean is NaN, first test fails and yields pos=2.
        if not len(selected):bolus[z]=2;continue
        mean=selected.mean(axis=0);pos=3
        while True:
            average=mean[:pos].mean()
            if abs((average-mean[pos])/average)<0.05:pos+=1
            else:pos-=1;break
            if pos==12:pos-=1;break
        bolus[z]=pos
        s0=selected[:,:pos].mean(axis=-1)
        baseline[z][support[z]]=s0
        concentration[z][support[z]]=-np.log(selected/s0[:,None])/te
    return concentration,baseline,bolus
