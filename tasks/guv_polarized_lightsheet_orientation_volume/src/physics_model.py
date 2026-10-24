"""Explicit polarized dipole/finite-NA angular-spatial acquisition operator.

Published Polaris mathematics adapted under MIT (see LICENSE). The original
odd-size reflection omission is corrected by absolute FFT-frequency indexing.
Angular products are evaluated with exact finite polynomial quadrature, rather
than loading the original precomputed Gaunt tensor or a domain library.
"""
from functools import lru_cache
import numpy as np
from scipy.special import sph_harm_y
LM=tuple((l,m) for l in (0,2,4) for m in range(-l,l+1))
VOXEL_NM=130.;WAVELENGTH_NM=561.;REFRACTIVE_INDEX=1.33
DETECTION_NA=(1.1,.67);AXIAL_SIGMA=.25

def real_harmonics(theta,phi,lmax=4):
    """Real orthonormal SH in lexicographic even l,m order, SH axis last."""
    values=[]
    for l,m in LM:
        if l>lmax:break
        y=sph_harm_y(l,m,theta,phi)
        if m>0:y=(y+(-1.)**m*sph_harm_y(l,-m,theta,phi))/np.sqrt(2)
        elif m<0:y=(y-(-1.)**m*sph_harm_y(l,-m,theta,phi))/(np.sqrt(2)*1j)
        values.append(y.real)
    return np.stack(values,axis=-1)

@lru_cache(maxsize=1)
def angular_coupling():
    """Return15x6x3 integrals of SH products for detector indices0,3,5.

    A12-point Gauss-Legendre rule in cos(theta) and32-point periodic azimuth
    exactly integrate the finite-degree products here (up to angular degree8).
    Other detector angular components are identically absent in this experiment.
    """
    u,w=np.polynomial.legendre.leggauss(12);phi=np.arange(32)*2*np.pi/32
    y=real_harmonics(np.arccos(u)[:,None],phi[None,:]).reshape(-1,15)
    weights=np.broadcast_to(w[:,None],(12,32)).ravel()*2*np.pi/32
    G=np.einsum('ni,nj,nk,n->ijk',y,y[:,:6],y[:,[0,3,5]],weights)
    # These are exact angular-selection zeros; remove quadrature roundoff only.
    G[np.abs(G)<1e-14]=0
    G.flags.writeable=False
    return G

def angular_product(illumination,detection):
    """Project a dipole illumination times the selected radial detector to15SH."""
    return np.einsum('ijk,...j,...k->...i',angular_coupling(),illumination,np.asarray(detection)[...,[0,3,5]])

def rotate_quadrupole(c):
    """Original90degree y-axis rotation for coefficients through l2."""
    R=np.array([[1,0,0,0,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0],[0,0,0,-.5,0,np.sqrt(3)/2],[0,0,0,0,1,0],[0,0,0,np.sqrt(3)/2,0,.5]])
    return np.asarray(c)@R.T

def detector_coefficients(radius,na,view):
    """Six SH coefficients of the unpolarized paraxial finite-NA detector."""
    radius=np.asarray(radius);r=np.minimum(np.abs(radius)/2,1)
    root=r*np.sqrt(1-r*r);angle=np.arccos(r)
    A1=2/np.pi*(angle-root);A2=2/np.pi*(angle-(3-2*(np.abs(radius)/2)**2)*root)
    alpha=na/REFRACTIVE_INDEX
    c=np.zeros(radius.shape+(6,));c[...,0]=A1+alpha**2/4*A2;c[...,3]=(-A1+alpha**2/2*A2)/np.sqrt(5)
    return rotate_quadrupole(c) if view==1 else c

def transfer_fields(pols,shape=(77,55,80)):
    """Return original factorized OTFs Hxy,Hyz(float32),Hx(float64),Hz(float32)."""
    p=np.asarray(pols);p=p/np.linalg.norm(p,axis=-1,keepdims=True)
    theta=np.arccos(np.clip(p[...,2],-1,1));phi=np.arctan2(p[...,1],p[...,0])
    illumination=real_harmonics(theta,phi,lmax=2);illumination[...,1:]*=.4
    results={}
    for view,na in enumerate(DETECTION_NA):
        frequencies=[np.fft.rfftfreq(n,d=VOXEL_NM)*WAVELENGTH_NM/na for n in shape]
        first,second=(frequencies[0],frequencies[1]) if view==0 else (frequencies[1],frequencies[2])
        radius=np.sqrt(first[:,None]**2+second[None,:]**2)
        detection=detector_coefficients(radius,na,view)
        illumination0=np.array([1,0,0,-1/np.sqrt(5),0,0])/np.sqrt(4*np.pi)
        if view==0:illumination0=rotate_quadrupole(illumination0)
        norm=angular_product(illumination0,detector_coefficients(np.asarray(0.),na,view))[0]
        H=np.stack([angular_product(c,detection)/norm for c in illumination[view]],axis=-1).astype(np.float32)
        H=H/np.max(np.abs(H))
        results['Hxy' if view==0 else 'Hyz']=H
        q=frequencies[2] if view==0 else frequencies[0]
        results['Hz' if view==0 else 'Hx']=np.exp(-q*q/(2*AXIAL_SIGMA**2),dtype=np.float32) if view==0 else np.exp(-q*q/(2*AXIAL_SIGMA**2))
    return results

def frequency_matrix(h,z,shape,positive_only=False):
    """Return H(X,Y,SH,measurement) at one rFFT z-bin; flatten P then view."""
    X,Y,_=shape
    ix=np.arange(X//2+1) if positive_only else np.abs(np.rint(np.fft.fftfreq(X)*X)).astype(int)
    iy=np.arange(Y//2+1) if positive_only else np.abs(np.rint(np.fft.fftfreq(Y)*Y)).astype(int)
    H0=h['Hz'][z]*h['Hxy'][ix[:,None],iy[None,:]]
    H1=h['Hx'][ix,None,None,None]*h['Hyz'][iy[None,:],z]
    return np.stack([H0,H1],axis=-1).reshape(len(ix),len(iy),15,-1)

def forward_linear(f,h):
    """Return unnormalized XYZPV intensity from complete signed spatial bins.

    Complex64 Fourier detector accumulation preserves the original precision.
    Absolute frequency indexing restores odd-grid conjugate coverage.
    """
    shape=f.shape[:3];P=h['Hxy'].shape[-1]
    F=np.fft.rfftn(f,axes=(0,1,2));G=np.empty(F.shape[:3]+(P,2),np.complex64)
    for z in range(F.shape[2]):
        H=frequency_matrix(h,z,shape)
        G[:,:,z]=np.einsum('xyjd,xyj->xyd',H,F[:,:,z]).reshape(shape[:2]+(P,2))
    # NumPy1.26 original FFT promoted complex64 to complex128; NumPy2 preserves
    # precision. Explicit promotion preserves the original float64 image/mean.
    return np.fft.irfftn(G.astype(np.complex128),s=shape,axes=(0,1,2))
