"""Explicit calibrated-velocity to spectral-energy processing.
Qopen mathematical conventions, copyright Tom Eulenfeld, MIT; see LICENSE.
"""
import numpy as np
from scipy import signal,fftpack
SAMPLE_RATE=20.;DENSITY=2700.;FREE_SURFACE=4.;VELOCITY=3400.
def load_data(path="data/raw_data.npz"):
    with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}
def smooth(values,samples=20,pad=True):
    w=np.bartlett(samples);w/=w.sum()
    if pad:values=np.r_[np.zeros((samples-1)//2),values,np.zeros(samples//2)]
    return np.convolve(w,values,'valid')
def filter_width(band):
    b,a=signal.iirfilter(2,np.array(band)/(SAMPLE_RATE*.5),btype='band',ftype='butter',output='ba')
    w,h=signal.freqz(b,a)
    return (w[1]-w[0])/2/np.pi*SAMPLE_RATE*np.sum(np.abs(h)**4)
def observed_energy(velocity,band):
    sos=signal.iirfilter(2,np.array(band)/(SAMPLE_RATE*.5),btype='band',ftype='butter',output='sos')
    energies=[]
    for row in velocity:
        detrended=signal.detrend(row,type='linear')
        filtered=signal.sosfilt(sos,signal.sosfilt(sos,detrended)[::-1])[::-1]
        hilbert=fftpack.hilbert(filtered)
        energies.append(DENSITY*(filtered**2+hilbert**2)/2/filter_width(band)/FREE_SURFACE)
    return np.sum(energies,axis=0)
def sample_index(time,start):
    value=(time-start)*SAMPLE_RATE
    return int(np.floor(value+.5))
def prepare_pair(velocity,band,distance,start,sonset):
    energy=observed_energy(velocity,band)
    noise=min(np.mean(energy[sample_index(a,start):sample_index(b,start)+1]) for a,b in [(-10,-5),(-5,0)])
    corrected=energy-noise;corrected[corrected<noise/100]=noise/100
    first=sample_index(sonset-2,start);last=sample_index(sonset+20,start)
    bulk=corrected[first:last+1];tbgrid=np.arange(len(bulk))/SAMPLE_RATE+distance/VELOCITY-2
    eb=np.mean(bulk);tb=np.sum(bulk*tbgrid)/np.sum(bulk)
    smoothed=smooth(corrected);cs=sample_index(sonset+20,start)
    crossing=np.flatnonzero(smoothed[cs:]<3*noise)
    offset=int(crossing[0]) if len(crossing) else len(smoothed)-cs
    end=min(sonset+150,sonset+20+offset/SAMPLE_RATE)
    if end-(sonset+20)<.05:return None
    ce=sample_index(end,start);ec=smoothed[cs:ce+1]
    tc=np.arange(len(ec))/SAMPLE_RATE+distance/VELOCITY+20
    return dict(energy=smoothed,unsmoothed=corrected,tcoda=tc,Ecoda=ec,tbulk=tb,Ebulk=eb,bulk_interval=(tbgrid[0],tbgrid[-1]),bulk_weight=len(bulk),distance=distance,noise=noise)
def prepare_band(data,event,band_index):
    pairs=[]
    for j in np.flatnonzero(data['valid'][event]):
        p=prepare_pair(data['velocity'][event,j],data['frequency_bands'][band_index],data['distance'][event,j],data['start_time'][event,j],data['s_onset'][event,j])
        if p is not None:p['station_slot']=j;pairs.append(p)
    return pairs
