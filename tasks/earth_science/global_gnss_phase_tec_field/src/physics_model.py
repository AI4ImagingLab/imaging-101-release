"""Single-shell phase-difference TEC geometry and explicit real SH operator.

Adapted from Padokhin/Vesnin MosGIM2 (MIT, 2023), qualified continuous
time weights. SI geometry; angles radians; TEC and its coefficients in TECu.
"""
import numpy as np
from math import factorial,pi,sqrt
from scipy.sparse import csr_matrix

EARTH_RADIUS = 6371000.0
SHELL_HEIGHT = 450000.0
DEGREE = 15
INTERVALS = 24
SIGMA0 = 0.02
SIGMA_REL = 0.005

def harmonic_orders(degree=DEGREE):
    """Return degree-major (degree, signed order), each length (degree+1)^2."""
    return (np.array([n for n in range(degree+1) for m in range(-n,n+1)]),
            np.array([m for n in range(degree+1) for m in range(-n,n+1)]))

def real_harmonics(longitude, colatitude, degree=DEGREE):
    """Real orthonormal SH by explicit associated-Legendre recurrence.

    P_m^m=(-1)^m(2m-1)!! sin(theta)^m; P_(m+1)^m=(2m+1)cos(theta)P_m^m;
    (n-m)P_n^m=(2n-1)cos(theta)P_(n-1)^m-(n+m-1)P_(n-2)^m.
    The Condon-Shortley phase is then combined with the author's real basis.
    """
    lon=np.asarray(longitude);colat=np.asarray(colatitude);x=np.cos(colat)
    sine=np.sin(colat);out=np.empty((len(lon),(degree+1)**2));diagonal=np.ones(len(lon))
    for m in range(degree+1):
        if m:diagonal=-(2*m-1)*sine*diagonal
        previous=None;current=diagonal
        for n in range(m,degree+1):
            if n==m:P=current
            elif n==m+1:P=(2*m+1)*x*current
            else:P=((2*n-1)*x*current-(n+m-1)*previous)/(n-m)
            norm=sqrt((2*n+1)/(4*pi)*factorial(n-m)/factorial(n+m))
            if m==0:out[:,n*n+n]=norm*P
            else:
                amplitude=sqrt(2)*(-1.)**m*norm*P
                out[:,n*n+n-m]=amplitude*np.sin(m*lon)
                out[:,n*n+n+m]=amplitude*np.cos(m*lon)
            previous,current=current,P
    return out

def slant_factor(elevation, height=SHELL_HEIGHT):
    """Spherical-shell obliquity, dimensionless, elevation in radians."""
    return 1/np.sqrt(1-(EARTH_RADIUS*np.cos(elevation)/(EARTH_RADIUS+height))**2)

def design_matrix(data, degree=DEGREE, intervals=INTERVALS, sigma0=SIGMA0):
    """CSR current-minus-reference operator and diagonal inverse noise variance.

    data contains 1-D time/mlt1/mcolat1/el and corresponding *_ref arrays.
    Coefficients flatten (hour node, harmonic), with continuous hourly weights.
    Duplicate entries, including an identical reference ray, sum in CSR.
    """
    count=len(data['time']);nc=(degree+1)**2
    cols=[];values=[]
    for suffix,sign in [('',1.),('_ref',-1.)]:
        u=np.asarray(data['time'+suffix])*intervals/86400
        node=np.floor(u).astype(int);fraction=u-node
        if np.any(node<0) or np.any(node>=intervals):
            raise ValueError('Observation times must lie in [0,86400) seconds')
        y=real_harmonics(data['mlt1'+suffix],data['mcolat1'+suffix],degree)
        y*=slant_factor(data['el'+suffix])[:,None]
        for offset,w in [(0,1-fraction),(1,fraction)]:
            cols.append((node[:,None]+offset)*nc+np.arange(nc)[None,:])
            values.append(sign*w[:,None]*y)
    # Interleave the four contributions per row, preserving source ordering.
    columns=np.stack(cols,axis=1).reshape(-1)
    val=np.stack(values,axis=1).reshape(-1)
    rows=np.repeat(np.arange(count),4*nc)
    A=csr_matrix((val,(rows,columns)),shape=(count,(intervals+1)*nc))
    s=np.sin(data['el'])**2;sr=np.sin(data['el_ref'])**2
    weight=(1/sigma0**2)*s*sr/(s+sr)
    return A,weight

def chunk_bounds(count, gigs=0.1):
    """Author's deterministic memory partition, including its final merge rule."""
    nchunks=max(1,int(count/(30000*gigs)))
    if nchunks==1:return [(0,count)]
    step=count//nchunks
    chunks=[(i-step,i) for i in range(step,count,step)]
    if (count-chunks[-1][1])/count > .1:chunks.append((chunks[-1][1],count))
    else:chunks[-1]=(chunks[-1][0],count)
    return chunks

def temporal_precision(degree=DEGREE, intervals=INTERVALS, sigma_rel=SIGMA_REL):
    """Dense first-difference precision; no periodic endpoint connection."""
    nc=(degree+1)**2;size=(intervals+1)*nc;R=np.zeros((size,size))
    for j in range(intervals):
        a=np.arange(nc)+j*nc;b=a+nc;w=1/sigma_rel**2
        R[a,a]+=w;R[b,b]+=w;R[a,b]-=w;R[b,a]-=w
    return R

def grid_basis(colatitude=None, longitude=None, degree=DEGREE):
    """Return (latitude*longitude, SH) basis, north-to-south then local angle."""
    if colatitude is None:colatitude=np.deg2rad(np.arange(2.5,180,2.5))
    if longitude is None:longitude=np.deg2rad(np.arange(0.,360.,5.))
    lon,colat=np.meshgrid(longitude,colatitude)
    return real_harmonics(lon.ravel(),colat.ravel(),degree)

def render_maps(coefficients, colatitude=None, longitude=None, degree=DEGREE):
    """Evaluate every hourly coefficient block; returns (hour,71,72) by default."""
    nlat=71 if colatitude is None else len(colatitude)
    nlon=72 if longitude is None else len(longitude)
    c=np.asarray(coefficients).reshape(-1,(degree+1)**2)
    return (c@grid_basis(colatitude,longitude,degree).T).reshape(len(c),nlat,nlon)

def predict(coefficients, data, chunk_size=4096):
    """Full-ray phase-derived slant TEC differences; no data or quality mask."""
    result=np.empty(len(data['rhs']))
    for start in range(0,len(result),chunk_size):
        end=min(start+chunk_size,len(result));d={k:v[start:end] for k,v in data.items()}
        A,_=design_matrix(d);result[start:end]=A@np.asarray(coefficients).ravel()
    return result

def residual_metrics(prediction,data):
    """Unweighted and elevation-weighted RMS, both in TECu."""
    e=np.asarray(prediction)-data['rhs'];s=np.sin(data['el'])**2;t=np.sin(data['el_ref'])**2
    w=s*t/(s+t)
    return {'rms_TECu':float(np.sqrt(np.mean(e**2))),
            'weighted_rms_TECu':float(np.sqrt(np.dot(w,e**2)/w.sum()))}
