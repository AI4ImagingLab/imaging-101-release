"""Explicit numerical extraction, developed against pre-frozen independent stages."""
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.fft import next_fast_len

def index(energy,offset):
    return int(np.rint((energy-offset)/.2))

def powerlaw(y,x,lo,hi):
    i,j=index(lo,x[0]),index(hi,x[0])
    if (i+j)%2:j-=1
    if i==j:j+=2
    m=(i+j)//2
    r=2*(np.log(y[i:m].sum())-np.log(y[m:j].sum()))/(np.log(x[j])-np.log(x[i]))
    k=1-r;A=k*y[m:j].sum()/(x[j]**k-x[m]**k)
    return A,r

def extend(y,x,window):
    A,r=powerlaw(y,x,x[-window],x[-1])
    xx=x[0]+.2*np.arange(len(x)+100)
    return np.r_[y,.2*A*xx[len(x):]**(-r)],xx

def taper(y):
    y=y.copy();n=max(20,round(.02*len(y)));h=np.hanning(2*n)
    y[:n]*=h[:n];y[-n:]*=h[-n:]
    return y

def solve(low,core):
    delta=(-40+.2*np.argmax(low))
    xl=-40-delta+.2*np.arange(len(low));xc=360-delta+.2*np.arange(len(core))
    A,r=powerlaw(core,xc,600.,638.)
    background=.2*A*xc**(-r)
    start=index(600.,xc[0]);edge=(core-background)[start:];xe=xc[start:]
    le,xle=extend(low,xl,100);ce,xce=extend(edge,xe,20)
    lt,ct=taper(le),taper(ce)
    imin,imax=index(1.,xl[0]),index(11.,xl[0])
    d=savgol_filter(low[imin:imax],5,3,deriv=1,delta=.2)
    threshold=xl[imin+np.argmin(abs(d))]
    I0=lt[:index(threshold,xl[0])].sum()
    roots=UnivariateSpline(xle,lt-lt.max()/2,s=0).roots();width=roots[1]-roots[0]
    n=next_fast_len(len(lt)+len(edge)-1)
    gx=xl[0]+.2*np.arange(n);sigma=width/2.3548
    g=np.exp(-.5*(gx/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    full=np.fft.irfft(np.fft.rfft(g)*np.fft.rfft(ct,n=n)/np.fft.rfft(lt,n=n),n=n)*I0
    return dict(background=background,background_A=A,background_r=r,edge=edge,core_extended=ce,low_extended=le,core_tapered=ct,low_tapered=lt,threshold=threshold,fwhm=width,I0=I0,reconstruction=full[:len(edge)],low_axis=xl,core_axis=xc,full_reconstruction=full,gaussian=g,nfft=n)

