"""BSD-3-Clause extraction of PyLorentz's Teague double-inverse-Laplacian TIE.
Copyright (c) PyLorentz contributors. The source has periodic centered stencils,
a circular bandwidth cutoff, and optional even extension/Tikhonov response.
"""
import numpy as np
from .preprocessing import prepare,DEFOCUS_NM,PIXEL_NM,VOLTAGE_V
from .physics_model import wavelength_nm,frequency_filters,periodic_gradient,even_extension,phase_to_induction
def reconstruct_phase(infocus,difference,defocus_nm,pixel_nm=PIXEL_NM,voltage=VOLTAGE_V,qc=0.,symmetric=False):
    """Invert one raw ±defocus intensity difference to a zero-minimum radian phase."""
    if not np.isfinite(qc) or qc<0:raise ValueError('qc must be finite nonnegative cycles/nm')
    shape=difference.shape
    if symmetric:infocus=even_extension(infocus);difference=even_extension(difference)
    inv,aperture=frequency_filters(difference.shape,pixel_nm,qc)
    potential=-np.fft.ifft2(np.fft.fft2(difference)*aperture*inv)
    gy,gx=periodic_gradient(potential)
    gy=np.real(gy/infocus);gx=np.real(gx/infocus)
    divergence=periodic_gradient(gy)[0]+periodic_gradient(gx)[1]
    prefactor=-pixel_nm**2/(16*np.pi**3*wavelength_nm(voltage)*defocus_nm)
    phase=np.real(prefactor*np.fft.ifft2(np.fft.fft2(divergence)*aperture*inv))
    if symmetric:phase=phase[:shape[0],:shape[1]]
    return phase-phase.min()
def reconstruct(unflipped,flipped,index=0,qc=0.,symmetric=False):
    """Return both radian phases and projected Bx,By in T nm, plus diagnostics."""
    stack,i0,db,de=prepare(unflipped,flipped,index)
    df=DEFOCUS_NM[4+index]
    if symmetric:
        ext=np.stack([even_extension(s) for s in stack])
        i0=even_extension(i0)
        db=.5*((ext[3]-ext[0])-(ext[4]-ext[1]));db-=db.mean()
        de=.5*((ext[3]-ext[0])+(ext[4]-ext[1]));de-=de.mean()
    pb=reconstruct_phase(i0,db,df,qc=qc)
    pe=reconstruct_phase(i0,de,df,qc=qc)
    if symmetric:
        ny,nx=unflipped.shape[1:]
        pb=pb[:ny,:nx];pb-=pb.min()
        pe=pe[:ny,:nx];pe-=pe.min()
        i0=i0[:ny,:nx];db=db[:ny,:nx];de=de[:ny,:nx]
    bx,by=phase_to_induction(pb)
    return dict(phase_B=pb,phase_E=pe,Bx=bx,By=by,infocus=i0,difference_B=db,difference_E=de)
