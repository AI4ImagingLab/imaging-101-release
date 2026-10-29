"""Local extraction of EISPY2D mathematical kernels.
Copyright (c) André Costa Batista and EISPY2D contributors.
Adapted from revision a72831db93d64c7b80282579f48de4c110627eeb.
Distributed under GNU General Public License version3; see LICENSE.
No upstream framework or computational-imaging library is imported.
"""

import numpy as np
from scipy.special import jv,hankel1,hankel2
from scipy.spatial.distance import cdist,pdist,squareform
GRID=30
LENGTH=.8
BACKGROUND_EPSILON=4.
WAVENUMBER=12.575070131717595+0j
RADIUS=1.
def coordinates(n=GRID):
    dx=LENGTH/n
    return np.meshgrid(np.arange(-LENGTH/2+.5*dx,LENGTH/2,dx),np.arange(-LENGTH/2+.5*dx,LENGTH/2,dx))
def green_matrices(receiver_angles,n=GRID):
    x,y=coordinates(n);dx=x[0,1]-x[0,0];dy=y[1,0]-y[0,0];a=np.sqrt(dx*dy/np.pi);k=WAVENUMBER
    receiver_x=RADIUS*np.cos(receiver_angles);receiver_y=RADIUS*np.sin(receiver_angles)
    cells=np.column_stack((x.ravel(),y.ravel()));receivers=np.column_stack((receiver_x,receiver_y))
    distance=cdist(receivers,cells);gs=-1j*np.pi*k*a/2*jv(1,k*a)*hankel2(0,k*distance)
    distance=squareform(pdist(cells));np.fill_diagonal(distance,1.)
    gd=-1j*np.pi*k*a/2*jv(1,k*a)*hankel2(0,k*distance)
    np.fill_diagonal(gd,-(1j/2)*(np.pi*k*a*hankel2(1,k*a)-2j))
    return gs,gd
def incident_field(illumination_angles,n=GRID):
    x,y=coordinates(n);phi=illumination_angles.reshape(1,-1)
    return np.exp(-1j*WAVENUMBER*(x.reshape(-1,1)@np.cos(phi)+y.reshape(-1,1)@np.sin(phi)))
def extended_kernel(n=GRID):
    dx=LENGTH/n;a=np.sqrt(dx*dx/np.pi);k=WAVENUMBER
    q=np.arange(-LENGTH/2-(n/2-1)*dx,LENGTH/2+n/2*dx,dx)
    xx,yy=np.meshgrid(q,q);r=np.sqrt(xx**2+yy**2);r[n-1,n-1]=1.
    raw=1j*np.pi*k*a/2*jv(1,k*a)*hankel1(0,k*r)
    raw[n-1,n-1]=1j*np.pi*k*a/2*hankel1(1,k*a)-1
    g=np.empty((2*n-1,2*n-1),dtype='complex128')
    g[:n,:n]=raw[n-1:2*n-1,n-1:2*n-1];g[n:,n:]=raw[:n-1,:n-1]
    g[n:,:n]=raw[:n-1,n-1:2*n-1];g[:n,n:]=raw[n-1:2*n-1,:n-1]
    return g
def prepare_fft(n=GRID,sources=10):
    g=extended_kernel(n);h=np.tile(g[:,:,None],(1,1,sources))
    return {'G':g,'fft':np.fft.fft2(h,axes=(0,1)),'adjoint_fft':np.fft.fft2(np.conj(h),axes=(0,1)),'n':n}
def fft_A(field,contrast,cache):
    n=cache['n'];sources=field.shape[1];u=(contrast.reshape(-1,1)*field).reshape(n,n,sources)
    e=np.fft.ifft2(cache['fft']*np.fft.fft2(u,axes=(0,1),s=(2*n-1,2*n-1)),axes=(0,1))[:n,:n]
    return field-e.reshape(n*n,sources)
def fft_AH(field,contrast,cache):
    n=cache['n'];sources=field.shape[1];u=field.reshape(n,n,sources)
    e=np.fft.ifft2(cache['adjoint_fft']*np.fft.fft2(u,axes=(0,1),s=(2*n-1,2*n-1)),axes=(0,1))[:n,:n]
    return field-np.conj(contrast.reshape(-1,1))*e.reshape(n*n,sources)
def physical_contrast(contrast):
    epsilon=BACKGROUND_EPSILON*(contrast.real+1)
    return (epsilon-0j)/(BACKGROUND_EPSILON-0j)-1
def cg_fft(rhs,contrast,cache,maximum_iterations=500,tolerance=1e-8):
    """Preserve the author's shared-denominator, per-source PR update."""
    old=np.zeros_like(rhs);current=old.copy();r=fft_A(old,contrast,cache)-rhs;gradient_old=fft_AH(r,contrast,cache);direction=-gradient_old
    errors=[];rhs_norm=np.linalg.norm(rhs)
    if rhs_norm==0:return old,np.array([0.])
    for iteration in range(maximum_iterations):
        ap=fft_A(direction,contrast,cache)
        denominator=np.linalg.norm(ap.reshape(-1,1,order='F'),ord='fro')**2
        if denominator==0:break
        alpha=-np.sum(np.conj(ap)*(fft_A(old,contrast,cache)-rhs),axis=0)/denominator
        current=old+np.tile(alpha,(rhs.shape[0],1))*direction
        r=fft_A(current,contrast,cache)-rhs;gradient=fft_AH(r,contrast,cache)
        error=np.linalg.norm(r)/rhs_norm;errors.append(error)
        if error<tolerance:break
        beta=np.sum(np.conj(gradient)*(gradient-gradient_old),axis=0)/np.sum(np.abs(gradient_old)**2,axis=0)
        direction=-gradient+np.tile(beta,(rhs.shape[0],1))*direction
        old=current.copy();gradient_old=gradient.copy()
    return current,np.asarray(errors)
def total_field(contrast,incident,cache=None):
    n=int(np.sqrt(contrast.size))
    if cache is None:cache=prepare_fft(n,incident.shape[1])
    internal,error=cg_fft(incident.conj(),physical_contrast(contrast),cache)
    return internal.conj(),error
def scattered_field(contrast,field,gs):return gs@(contrast.reshape(-1,1)*field)
def forward(contrast,illumination_angles,receiver_angles):
    n=int(np.sqrt(contrast.size));gs,_=green_matrices(receiver_angles,n);incident=incident_field(illumination_angles,n)
    field,error=total_field(contrast,incident)
    return scattered_field(physical_contrast(contrast),field,gs),field,error
