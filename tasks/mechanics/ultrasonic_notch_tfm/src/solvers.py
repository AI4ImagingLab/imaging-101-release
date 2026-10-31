"""Explicit delay-and-sum with nearest-sample interpolation."""
import numpy as np
from .preprocessing import analytic_filter,DT
from .physics_model import reconstruction_grid,travel_times,reciprocity_weights
def delay_and_sum(filtered,tx,rx,times,t0=0.,dt=DT,chunk=256):
    ntrace,nsample=filtered.shape
    out=np.zeros(len(times),dtype=np.complex128); w=reciprocity_weights(tx,rx)
    for lo in range(0,len(times),chunk):
        tau=times[lo:lo+chunk,tx]+times[lo:lo+chunk,rx]
        ind=np.rint((tau-t0)*(1./dt)).astype(np.int64)
        valid=(ind>=0)&(ind<nsample)
        values=filtered[np.arange(ntrace)[None,:],np.clip(ind,0,nsample-1)]
        out[lo:lo+chunk]=np.sum(values*valid*w,axis=1)/ntrace
    return out
def reconstruct(data,speed=6300.,channel_step=1,pulse_echo=False):
    x,z=reconstruction_grid(); sel=(data['tx']%channel_step==0)&(data['rx']%channel_step==0)
    if pulse_echo: sel &= data['tx']==data['rx']
    traces=analytic_filter(data['traces'][sel]); times=travel_times(data['positions'],x,z,speed)
    return np.abs(delay_and_sum(traces,data['tx'][sel],data['rx'][sel],times,data['time'][0])).reshape(len(z),len(x))
