"""Explicit equal-area annular-sector eclipse integration and real harmonics.
Derived from SPIDERMAN mathematical conventions (Louden/Kreidberg), GPL3.
Independent angular quadrature evaluates the same constant-brightness tiles.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
def harmonics(lat,lon):
    s=np.sin(lat);c=np.cos(lat);q=np.sqrt(3);p=np.sqrt(15)
    return np.stack([np.ones_like(s),q*c*np.sin(lon),q*s,q*c*np.cos(lon),p/2*c*c*np.sin(2*lon),p*s*c*np.sin(lon),np.sqrt(5)/2*(3*s*s-1),p*s*c*np.cos(lon),p/2*c*c*np.cos(2*lon)],axis=-1)
def cells(n=20):
    low=[];high=[];start=[];end=[];mx=[];my=[]
    for i in range(n):
        count=2*i+1;step=2*np.pi/count
        for j in range(count):
            l=i/n;h=(i+1)/n;a=j*step;b=(j+1)*step;mid=0. if i==0 else (l+h)/2
            low.append(l);high.append(h);start.append(a);end.append(b);mx.append(mid*np.cos((a+b)/2));my.append(mid*np.sin((a+b)/2))
    return [np.array(x) for x in [low,high,start,end,mx,my]]
def visible_areas(x,y,radius=1/.155313,n=20):
    lo,hi,start,end,_,_=cells(n);area=.5*(hi**2-lo**2)*(end-start);distance=np.hypot(x,y)
    if distance>=radius+1:return area
    if distance+1<=radius:return np.zeros_like(area)
    angle=np.arctan2(y,x);nodes,weights=leggauss(64);result=[]
    for a,b,l,h,full in zip(start,end,lo,hi,area):
        events=[]
        for r in [l,h]:
            if r>0:
                cosine=(distance**2+r**2-radius**2)/(2*distance*r)
                if abs(cosine)<=1:events.extend([angle-np.arccos(cosine),angle+np.arccos(cosine)])
        if distance>radius:
            v=np.arcsin(radius/distance);events.extend([angle-v,angle+v])
        events=np.mod(events,2*np.pi);cuts=np.r_[a,np.sort(events[(events>a)&(events<b)]),b];blocked=0.
        for u,v in zip(cuts[:-1],cuts[1:]):
            psi=np.pi/4*(nodes+1);theta=u+(v-u)*np.sin(psi)**2;dot=x*np.cos(theta)+y*np.sin(theta);disc=radius**2-distance**2+dot**2;root=np.sqrt(np.maximum(disc,0));lower=np.maximum(l,np.maximum(0,dot-root));upper=np.minimum(h,dot+root);integrand=np.where(disc>=0,np.maximum(upper**2-lower**2,0),0);integrand=np.where(upper>=lower,integrand,0);blocked+=np.pi*(v-u)/8*np.sum(weights*np.sin(2*psi)*integrand)
        result.append(full-blocked)
    return np.array(result)
def geometry(time):
    period=2.21857567;t0=-2.21857/2;inc=np.deg2rad(85.71);scale=8.863/.155313;f=2*np.pi*(time-t0)/period
    def pos(t):
        ff=2*np.pi*(t-t0)/period;return scale*np.sin(ff),-scale*np.cos(ff)*np.cos(inc),scale*np.cos(ff)*np.sin(inc)
    x,y,z=pos(time);phase=np.mod((time-t0)/period,1);lam=np.mod(np.pi+2*np.pi*phase,2*np.pi);phi=np.arctan(y/z)
    lag=(.0313*1.4960e11*np.sin(inc)*(1-np.cos(f)))/299792458/86400
    return (*pos(time-lag),lam,phi)
def basis(times):
    lo,hi,start,end,mx,my=cells();rad=np.sqrt(1-mx**2-my**2);out=[];areas=[]
    for t in times:
        x,y,z,lam,phi=geometry(t);latitude=np.arcsin(rad*np.sin(phi)+my*np.cos(phi));longitude=-(lam+np.arctan2(mx,rad*np.cos(phi)-my*np.sin(phi)));H=harmonics(latitude,longitude);a=visible_areas(x,y) if z<0 else .5*(hi**2-lo**2)*(end-start);areas.append(a);out.append(a@H)
    return np.array(out).T,np.array(areas)

def paired_lightcurves(times):
    s,areas=basis(times);lc=np.empty((17,len(times)));lc[0]=1+s[0]
    for i in range(1,9):lc[2*i-1]=1-s[i];lc[2*i]=1+s[i]
    return lc,areas

def eigencurves(lightcurves):
    """Original centered covariance eigenbasis, with fixed harmless sign gauge."""
    M=lightcurves[1:]-lightcurves[1:].mean(axis=1,keepdims=True);latent,coeff=np.linalg.eig(np.cov(M));order=np.argsort(latent.real)[::-1];latent=latent[order].real;coeff=coeff[:,order].real
    for i in range(16):
        j=np.argmax(abs(coeff[:,i]))
        if coeff[j,i]<0:coeff[:,i]*=-1
    return coeff,coeff.T@M,latent

def design_matrix(lightcurves,scores,components=3):
    return np.column_stack([lightcurves[0]-1,np.ones(lightcurves.shape[1]),scores[:components].T])

def spherical_coefficients(parameters,pca_coefficients):
    """Linear map from [uniform,stellar offset,PC amplitudes] to nine harmonics."""
    parameters=np.asarray(parameters);n=parameters.shape[-1]-2;co=np.zeros(parameters.shape[:-1]+(9,));co[...,0]=parameters[...,0]
    co[...,1:]=parameters[...,2:]@(pca_coefficients[1::2,:n]-pca_coefficients[0::2,:n]).T
    return co

def brightness(coefficients,latitude,longitude):return np.einsum('...k,...k->...',harmonics(latitude,longitude),coefficients)
