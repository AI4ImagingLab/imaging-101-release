"""Explicit azimuthal Fourier/latitude eigenfunction potential inversion.
Adapted from pfsspy, copyright Yeates2017/contributors2018 onward; GPL3.
"""
import numpy as np
from .physics_model import grid,field_from_potential
def solve(magnetogram,nr=35,rss=2.5,return_diagnostics=False):
    """Balanced float64(ns,nphi) Gauss map -> three face-centered fields.
    No domain package, optimizer or reference arrays enter this operator.
    """
    br=np.asarray(magnetogram,dtype=np.float64);ns,nphi=br.shape;g=grid(ns,nphi,nr,rss)
    ds,dp,dr=g['ds'],g['dp'],g['dr'];sg,sc=g['sg'],g['sc'];k=np.linspace(0,nr,nr+1)
    Fp=np.zeros(ns+1);Fp[1:-1]=np.sqrt(1-sg[1:-1]**2)/(np.arcsin(sc[1:])-np.arcsin(sc[:-1]))*dp
    Fs=(np.arcsin(sg[1:])-np.arcsin(sg[:-1]))/np.sqrt(1-sc**2)/dp
    Vg=Fp/ds/dp;Uc=Fs/ds/dp
    boundary=np.fft.rfft(br,axis=1).astype(np.complex128)
    A=np.diag(-Vg[1:-1],1)+np.diag(-Vg[1:-1],-1)
    mu=4*np.sin(np.pi*np.fft.fftfreq(nphi))**2
    psir=np.zeros((nr+1,ns),dtype=np.complex128);psi=np.zeros((nr+1,ns,nphi),dtype=np.complex128)
    e1=np.exp(dr);fact=np.sinh(dr)*(e1-1)
    for m in range(nphi//2+1):
        np.fill_diagonal(A,Vg[:-1]+Vg[1:]+Uc*mu[m])
        lam,Q=np.linalg.eigh(A);Q=Q.astype(np.complex128)
        F=.5*(1+e1+lam*fact);plus=F+np.sqrt(F*F-e1);minus=e1/plus
        for l in range(ns):
            if l==0 and m==0:continue
            cd=np.dot(Q[:,l],np.asfortranarray(boundary[:,m]))/lam[l]
            ratio=(minus[l]**(nr-1)-minus[l]**nr)/(plus[l]**nr-plus[l]**(nr-1))
            d=cd/(1+ratio);c=ratio*d
            psir[:,l]=c*plus[l]**k+d*minus[l]**k
        psi[:,:,m]=np.dot(psir,Q.T)
        if m>0:psi[:,:,nphi-m]=np.conj(psi[:,:,m])
    psi=np.fft.ifft(psi,axis=2).real
    als=np.zeros((nphi+1,ns,nr+1));alp=np.zeros((nphi,ns+1,nr+1))
    for j in range(nr+1):
        for i in range(nphi+1):als[i,:,j]=Fs*(psi[j,:,(i-1)%nphi]-psi[j,:,i%nphi])
        for i in range(nphi):alp[i,1:-1,j]=Fp[1:-1]*(psi[j,1:,i]-psi[j,:-1,i])
    fields=field_from_potential(als,alp,g)
    if return_diagnostics:return fields,dict(A=A,Fs=Fs,Fp=Fp,mu=mu,psi_probe=psi[::7,::15,::30],als_probe=als[::30,::15,::7],alp_probe=alp[::30,::15,::7])
    return fields
