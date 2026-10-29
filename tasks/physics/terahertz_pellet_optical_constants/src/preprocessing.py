"""Measured waveform loading and time-preserving Hann windows.
Adapted from THzPy, LGPL-3.0; source credit and license in validation.md/LICENSE.
"""
import numpy as np
THICKNESSES_MM=np.array([2.56,2.65])
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False)as f:t=f['time_ps'].copy();e=f['waveforms'].copy()
    for x in (t,e):
        if x.shape!=(1,3,3008)or x.dtype.kind!='f'or not np.isfinite(x).all():raise ValueError('Expected finite real (1,3,3008)')
    if not (np.diff(t,axis=-1)>0).all():raise ValueError('Time must increase')
    return t[0],e[0]
def common_window(time_ps,fields,half_width=15.,window='hanning'):
    """Return (3,2,N) field,time with original absolute delays preserved.
    Input channels: mixture, PE, air; times ps and field arbitrary units.
    """
    times=np.asarray(time_ps,dtype=np.float64);fields=np.asarray(fields,dtype=np.float64)
    peaks=np.argmax(abs(fields),axis=1);peak_times=times[np.arange(3),peaks];windows=[]
    for t,e,peak in zip(times,fields,peaks):
        dt=np.mean(np.diff(t));n=round(half_width/dt);length=2*n
        left=int(max(length/2-peak,0));right=int(max(length/2-(len(e)-peak),0))
        a=int(peak-length/2)+left;b=int(peak+length/2)+left
        taper=np.hanning(length)if window=='hanning'else np.ones(length)
        if window not in ('hanning','boxcar'):raise ValueError('Unsupported taper')
        windows.append((np.pad(e,(left,right))[a:b]*taper,dt*np.arange(-n,n)+t[peak]))
    latest=int(np.argmax(peak_times));time=windows[latest][1];dt=np.mean(np.diff(time));n=len(time)
    max_pad=round((max(peak_times)-min(peak_times))/dt)+1
    time=np.concatenate([dt*np.arange(-max_pad,0)+time[0],time]);result=[]
    for i,(e,t)in enumerate(windows):
        if len(e)!=n:e=np.interp(np.arange(n),np.arange(len(e)),e)
        right=round((peak_times[latest]-peak_times[i])/dt)
        result.append(np.array([np.pad(e,(max_pad-right,right)),time]))
    return np.array(result)
