"""Explicit 3D radiative-transfer Green function (Paasschens approximation).
Adapted from Qopen rt.py, copyright Tom Eulenfeld, MIT; see LICENSE.
"""
import numpy as np
from .preprocessing import VELOCITY,smooth
def coda_green(distance,time,scattering):
    r=distance*scattering;t=time*VELOCITY*scattering
    a=1-r**2/t**2
    return a**.125/(4*np.pi*t/3)**1.5*np.exp(t*(a**.75-1))*np.sqrt(1+2.026/(t*a**.75))*scattering**3
def green(distance,time,scattering):
    time=np.asarray(time);g=np.zeros(len(time));dt=float(time[1]-time[0]);i=np.count_nonzero(VELOCITY*time-distance<0)
    g[i+1:]=coda_green(distance,time[i+1:],scattering)
    if 0<i<len(g):
        arrival=distance/VELOCITY
        g[i]=np.exp(-VELOCITY*arrival*scattering)/(4*np.pi*VELOCITY**2*arrival**2)/dt/VELOCITY
    return g
def smoothed_green(distance,time,scattering):
    dt=time[1]-time[0];samples=int(round(1./dt));left=(samples-1)//2;right=samples//2
    extended=np.r_[time[0]-left*dt+np.arange(left)*dt,time,time[-1]+dt+np.arange(right)*dt]
    return smooth(green(distance,extended,scattering),samples,pad=False)
def green_observations(pairs,scattering):
    coda=[smoothed_green(p['distance'],p['tcoda'],scattering) for p in pairs]
    bulk=[np.mean(green(p['distance'],np.linspace(*p['bulk_interval'],1000),scattering)) for p in pairs]
    return coda,bulk
