"""Explicit LSCF polynomial normal equations and LSFD residue extraction.
Adapted from SDyPy-EMA numerical routines; MIT, task LICENSE.
"""
import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import least_squares
from .physics_model import pole_properties,modal_design
APPROXIMATE_MODES=[176,476,932,1534,2258,3161,4180]
def prepare(frequency,accelerance,upper=5000):
    end=np.argmin(abs(frequency-upper));f=frequency[:end];h=accelerance[:,:end]
    dt=1/(2*f[-1]);start=np.arange(0,f[0],f[1]-f[0]);f=np.r_[start,f];h=np.column_stack([np.zeros((len(h),len(start))),h])
    return f,h,dt
def adjusted_irfft(values,lower,indices):
    nf=2*(values.shape[1]-1)
    return np.fft.irfft(values,n=nf)[:,indices]*nf-np.fft.irfft(values[:,:lower],n=nf)[:,indices]*nf
def identify_poles(frequency,frf,dt,order=60,lower=10):
    low=np.argmin(abs(frequency-lower));n=2*order;nf=2*(frf.shape[1]-1)
    sk=-adjusted_irfft(frf,low,np.arange(-n,n+1));tv=adjusted_irfft(frf.real**2+frf.imag**2,low,np.arange(n+1))
    rv=-np.fft.irfft(np.ones(low),n=nf)[np.arange(n+1)]*nf;rv[0]+=nf
    s=[toeplitz(row[n:],row[:n+1][::-1]) for row in sk];t=toeplitz(tv.sum(axis=0));r=toeplitz(rv);allp=[]
    for j in range(2,n+1,2):
        inv=np.linalg.inv(r[:j+1,:j+1]);d=0
        for si in s:d-=si[:j+1,:j+1].T@inv@si[:j+1,:j+1]
        d+=t[:j+1,:j+1]
        coeff=np.linalg.solve(-d[:j,:j],d[:j,j]);allp.append(-np.log(np.roots(np.r_[coeff,1][::-1]))/dt)
    return allp
def stabilization(allp,frequency_tolerance=.001,damping_tolerance=.05):
    nmax=len(allp);ft=np.zeros((2*nmax,nmax));xt=np.zeros_like(ft);tf=np.zeros_like(ft);tx=np.zeros_like(ft)
    # Preserve the author's order-index convention (n=1 is the seed order).
    with np.errstate(divide='ignore',invalid='ignore'):
        for n,p in enumerate(allp):
            f,x=pole_properties(p);keep=np.ones(len(f),bool)
            for i in range(len(f)):
                if np.any(abs(f[i+1:]-f[i])<1e-3):keep[i]=False
            f=f[keep];x=x[keep]
            if n==1:ft[:len(f),0]=f;xt[:len(x),0]=x
            else:
                ft[:,n-2][ft[:,n-2]==0]=1e-10;xt[:,n-2][xt[:,n-2]==0]=1e-10
                for i,(ff,xx) in enumerate(zip(f,x)):
                    tf[i,n-1]=np.sum((abs((ff-ft[:,n-2])/ft[:,n-2])<frequency_tolerance)[:2*n]);tx[i,n-1]=np.sum((abs((xx-xt[:,n-2])/xt[:,n-2])<damping_tolerance)[:2*n])
                    ft[i,n-1]=ff;xt[i,n-1]=xx
    return ft,xt,tf,tx
def select_poles(allp,approximate=APPROXIMATE_MODES):
    ft,xt,tf,tx=stabilization(allp);mask=(tf>0)&(tx>0)&(xt>0);stable=np.where(mask,ft,0);chosen=[];freq=[];damp=[]
    for fr0 in approximate:
        fr=float(fr0)
        for width in [50//i for i in range(2,100) if 50//i>3]+[2]:
            values=stable[(stable>fr-width)&(stable<fr+width)]
            fr=least_squares(lambda z:values.flatten()-z[0],[fr]).x[0]
        ix=np.unravel_index(np.argmin(abs(stable-fr)),stable.shape);order=ix[1]
        target=-xt[ix]*2*np.pi*ft[ix]+1j*2*np.pi*ft[ix]*np.sqrt(1-xt[ix]**2)
        chosen.append(allp[order][np.argmin(abs(allp[order]-target))]);freq.append(ft[ix]);damp.append(xt[ix])
    return np.array(chosen),np.array(freq),np.array(damp)
def fit_residues(poles,frequency,frf,lower=10,upper=5000):
    p=modal_design(poles,frequency);y=np.vstack([frf.T.real,frf.T.imag]);n=len(frequency)
    lo=np.argmin(abs(frequency-lower));hi=np.argmin(abs(frequency-upper));mask=np.zeros(2*n,bool);mask[lo:hi]=True;mask[n+lo:n+hi]=True
    coeff=np.linalg.lstsq(p[mask],y[mask],rcond=None)[0].T;m=len(poles);constants=coeff[:,:m]+1j*coeff[:,m:2*m]
    z=p@coeff.T;fit=(z[:n]+1j*z[n:]).T
    return constants,fit,coeff[:,-4]+1j*coeff[:,-3],coeff[:,-2]+1j*coeff[:,-1]
def reconstruct(data,order=60,upper=5000):
    f,h,dt=prepare(data['frequency'],data['accelerance'],upper);allp=identify_poles(f,h,dt,order);p,_,_=select_poles(allp);fn,xi=pole_properties(p);a,fit,lr,ur=fit_residues(p,f,h,upper=upper)
    return {'natural_frequency':fn[None,:],'damping_ratio':xi[None,:],'reconstruction':a[None,:],'fitted_accelerance':fit[None,:],'lower_residual':lr[None,:],'upper_residual':ur[None,:]}
