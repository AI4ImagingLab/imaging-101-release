"""Local Schmidt harmonic potential and its physical magnetic gradient.
Adapted from pymaginverse, MIT, Frenk Out et al. Explicit radial-gradient
and geodetic-flattening corrections are documented in test/validation.md.
"""
import math
import numpy as np
from scipy.special import lpmv
from scipy.interpolate import BSpline
from .preprocessing import geodetic_to_geocentric
REFERENCE_RADIUS_KM=6371.2
DEGREE=3
KNOTS=np.arange(-2300.,2400.,100.)
EPOCHS=np.array([-1500.,-500.,500.,1000.,1700.])
LATITUDES=np.linspace(-87.5,87.5,36)
LONGITUDES=np.arange(-177.5,180.,5.)

def harmonic_basis(coords,degree=DEGREE):
    """coords(N,5)=(theta,longitude,radius,cd,sd); return geodetic NED (N,3,C)."""
    theta,phi,r,cd,sd=np.asarray(coords).T
    x=np.cos(theta); st=np.sin(theta); cols=[]
    for n in range(1,degree+1):
        radial=(REFERENCE_RADIUS_KM/r)**(n+2)
        for m in range(n+1):
            scale=(-1)**m*math.sqrt((2 if m else 1)*math.factorial(n-m)/math.factorial(n+m))
            p=lpmv(m,n,x)
            pm=lpmv(m,n-1,x) if m<n else np.zeros_like(x)
            dp=-(n*x*p-(n+m)*pm)/(x*x-1)*st*scale
            p=p*scale
            for trig,other,sign in ([(np.cos(m*phi),np.sin(m*phi),1)] if m==0 else [(np.cos(m*phi),np.sin(m*phi),1),(np.sin(m*phi),np.cos(m*phi),-1)]):
                north=radial*trig*dp
                east=radial*m/st*other*p*sign
                down=-radial*(n+1)*trig*p
                cols.append(np.stack((north*cd+down*sd,east,down*cd-north*sd),axis=1))
    return np.stack(cols,axis=2)

def temporal_basis(time):
    """Cubic B-spline basis (N,43) on knots -2300 through2300 years CE."""
    return BSpline.design_matrix(time,KNOTS,3).toarray()

def observables(coefficients,basis,temporal):
    """Return physical XYZ,H,F,I,D; angles radians (N,7)."""
    xyz=np.einsum('njk,nk->nj',basis,temporal@coefficients)
    h=np.linalg.norm(xyz[:,:2],axis=1); f=np.linalg.norm(xyz,axis=1)
    return np.column_stack((xyz,h,f,np.arcsin(xyz[:,2]/f),np.arctan2(xyz[:,1],xyz[:,0])))

def observable_jacobian(basis,prediction,kind):
    """Derivative of each scalar observation w.r.t. instantaneous Gauss coefficients."""
    x,y,z,h,f=prediction[:,:5].T
    dh=(x[:,None]*basis[:,0]+y[:,None]*basis[:,1])/h[:,None]
    df=(h[:,None]*dh+z[:,None]*basis[:,2])/f[:,None]
    di=(h[:,None]*basis[:,2]-z[:,None]*dh)/f[:,None]**2
    dd=(x[:,None]*basis[:,1]-y[:,None]*basis[:,0])/h[:,None]**2
    return np.where((kind==4)[:,None],df,np.where((kind==5)[:,None],di,dd))

def field_maps(coefficients):
    """Geodetic north/east/down field, nT, shape(5,3,36,72)."""
    lon,lat=np.meshgrid(LONGITUDES,LATITUDES)
    theta,r,cd,sd=geodetic_to_geocentric(np.deg2rad(lat.ravel()))
    coords=np.column_stack((theta,np.deg2rad(lon.ravel()),r,cd,sd))
    b=harmonic_basis(coords); c=temporal_basis(EPOCHS)@coefficients
    return np.einsum('njk,tk->tjn',b,c).reshape(len(EPOCHS),3,len(LATITUDES),len(LONGITUDES))

