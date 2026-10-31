"""Physical-field review metrics and common figure conventions."""
import numpy as np
import matplotlib.pyplot as plt
def metrics(x,y):
    """Cosine NCC and reference-dynamic-range NRMSE without alignment."""
    x=np.asarray(x).ravel();y=np.asarray(y).ravel()
    return {'ncc':float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),'nrmse':float(np.sqrt(np.mean((x-y)**2))/np.ptp(y))}
def field_panel(ax,field,times,title,unit,limit=None,signed=False):
    """Display complete time-floor field; explicit units and no evaluation crop."""
    if signed:
        bound=max(float(np.max(np.abs(field))),np.finfo(float).tiny);limits=(-bound,bound)
    else:limits=(None,None) if limit is None else limit
    image=ax.imshow(field.T,origin='lower',aspect='auto',extent=[times[0],times[-1],.5,field.shape[1]+.5],cmap='coolwarm' if signed else 'viridis',vmin=limits[0],vmax=limits[1])
    ax.set(title=title,xlabel='time (s)',ylabel='floor index');plt.colorbar(image,ax=ax,label=unit)

def welch_power(values,dt):
    """Explicit one-sided Welch power spectrum; Hann/50% overlap, no detrend.

    Window lengthN//8, FFT nextpower2, normalization by squared window sum.
    This labels power units rather than silently calling MATLAB 'power' PSD.
    """
    values=np.asarray(values);length=len(values)//8;hop=length//2
    window=np.hanning(length);nfft=1<<int(np.ceil(np.log2(length)))
    powers=[np.abs(np.fft.rfft(values[start:start+length]*window,n=nfft))**2 for start in range(0,len(values)-length+1,hop)]
    power=np.mean(powers,axis=0)/np.sum(window)**2
    power[1:-1]*=2
    return np.fft.rfftfreq(nfft,dt),power
