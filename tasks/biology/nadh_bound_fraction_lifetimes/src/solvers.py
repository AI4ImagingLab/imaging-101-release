"""Local explicit original two-harmonic semicircle graphical search (MIT).

Adapted from PhasorPy0.12 _lifetime_search_2, theory Vallmitjana etal2020.
No domain reconstruction library is imported. Candidate operations are vectorized
over pixels, preserving original candidate order and strict minimum selection.
"""
import numpy as np
from .physics_model import OMEGA,calibrated_phasors
def graphical_search(z,lifetime_range=(.2,.8,.01)):
    shape=z.shape[1:];r=z.real.reshape(2,-1);s=z.imag.reshape(2,-1);n=r.shape[1]
    active=np.isfinite(r).all(0)&np.isfinite(s).all(0)&(r>=0).all(0)&(r<=1).all(0)&(s>=0).all(0)&(s*s<=r-r*r+1e-9).all(0)
    g0=np.full(n,np.nan);g1=g0.copy();frac=g0.copy();best=np.full(n,np.inf)
    start,stop,step=lifetime_range
    grid=start+step*np.arange(int(round((stop-start)/step))+1)
    candidates=1/(1+(OMEGA*grid)**2)
    for candidate in candidates:
        a=candidate;b=np.sqrt(a-a*a);dx=r[0]-a;dy=s[0]-b;dr=dx*dx+dy*dy;dd=(a-.5)*s[0]-(r[0]-.5)*b;disc=.25*dr-dd*dd
        degenerate=active&((disc<0)|(dr<=0));g0[degenerate]=a;g1[degenerate]=a;frac[degenerate]=1.;active[degenerate]=False
        safe=np.where(dr>0,dr,1);root=np.sqrt(np.maximum(disc,0))
        aa=(dd*dy-np.copysign(1.,dy)*dx*root)/safe+.5;bb=(-dd*dx-abs(dy)*root)/safe
        cc=(dd*dy+np.copysign(1.,dy)*dx*root)/safe+.5;ee=(-dd*dx+abs(dy)*root)/safe
        swap=aa<cc;aa,cc=np.where(swap,cc,aa),np.where(swap,aa,cc);bb,ee=np.where(swap,ee,bb),np.where(swap,bb,ee)
        good=active&(bb>=0)&(ee>=0)
        ha=aa/(4-3*aa);hc=cc/(4-3*cc);good&=(ha>=0)&(ha<=1)&(hc>=0)&(hc<=1)
        hb=np.sqrt(np.maximum(ha-ha*ha,0));he=np.sqrt(np.maximum(hc-hc*hc,0));dx=hc-ha;dy=he-hb;x=r[1]-ha;y=s[1]-hb;length=dx*dx+dy*dy
        t=(x*dx+y*dy)/np.where(length!=0,length,1);dist=(x-t*dx)**2+(y-t*dy)**2;take=good&(length!=0)&(dist<best)
        best[take]=dist[take];g0[take]=aa[take];g1[take]=cc[take];frac[take]=t[take]
    g=np.stack([g0,g1]);tau=np.sqrt((1-g)/(g*OMEGA*OMEGA))
    return tau.reshape((2,)+shape),np.stack([1-frac,frac]).reshape((2,)+shape)
def single_harmonic(z,mean,free_lifetime=.4):
    a=1/(1-1j*OMEGA*free_lifetime);valid=np.isfinite(z);w=np.where(valid,mean,0.);center=np.nansum(z*w)/w.sum();d=center-a
    t=-2*np.real(np.conj(a-.5)*d)/abs(d)**2;b=a+t*d
    fraction=np.real((z-a)*np.conj(b-a))/abs(b-a)**2
    return np.clip(fraction,0,1),float(np.sqrt((1-b.real)/(b.real*OMEGA**2)))
def reconstruct(signal,reference,lifetime_range=(.2,.8,.01)):
    mean,z=calibrated_phasors(signal,reference);tau,fraction=graphical_search(z,lifetime_range)
    single,bound=single_harmonic(z[0],mean)
    return dict(reconstruction=np.nan_to_num(tau)[None],bound_fraction=np.nan_to_num(fraction[1])[None]),dict(mean=mean,phasors=z,fractions=fraction,single_fraction=np.nan_to_num(single),single_bound_lifetime=bound)
