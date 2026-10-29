"""Native clocks, author edge rejection and explicit biharmonic gap equations."""
from pathlib import Path
import numpy as np
from scipy.sparse import diags,kronsum
from scipy.sparse.linalg import spsolve
from .physics_model import Spectrum

def reflecting_laplacian(shape):
    """2D five-point discrete Laplacian with half-sample reflecting boundary."""
    nf,nt=shape
    def axis(n):
        diagonal=np.full(n,-2.);diagonal[[0,-1]]=-1.
        return diags([np.ones(n-1),diagonal,np.ones(n-1)],[-1,0,1],shape=(n,n),format='csr')
    return kronsum(axis(nt),axis(nf),format='csr')

def fill_gaps(intensity,method='biharmonic'):
    """Source zeros/NaNs are flagged; solve Δ²u=0 then preserve known fluxes.

    Matrix rows for unknown samples equal zero, with all known samples held
    fixed. The original inpainting routine clips its result to known intensity
    limits; that postprocessing is preserved explicitly. Mean refill is used
    only for the original all-eight temporal stitching stage.
    """
    result=np.array(intensity,dtype=np.float64,copy=True)
    missing=(result==0)|~np.isfinite(result)
    if not np.any(missing):return result
    if np.all(missing):raise ValueError('No valid intensity samples')
    known=result[~missing]
    if method=='mean':result[missing]=np.mean(known);return result
    if method!='biharmonic':raise ValueError('Unsupported selected gap protocol')
    laplacian=reflecting_laplacian(result.shape)
    biharmonic=(laplacian@laplacian).tocsr()
    unknown=np.flatnonzero(missing.ravel())
    fixed=result.ravel().copy();fixed[unknown]=0.
    rows=biharmonic[unknown]
    rhs=-rows@fixed
    matrix=rows[:,unknown].tocsc()
    solution=spsolve(matrix,rhs)
    result.ravel()[unknown]=np.clip(solution,np.min(known),np.max(known))
    return result

def make_spectrum(intensity,frequencies,times,mjd):
    """Load complete native arrays and actively remove only leading short subints."""
    d=Spectrum();d.dyn=np.array(intensity,dtype=np.float64,copy=True)
    d.freqs=np.array(frequencies,dtype=np.float64,copy=True);d.times=np.array(times,dtype=np.float64,copy=True)
    d.mjd=float(mjd)+d.times[0]/86400.;d.times-=d.times[0]
    d.nchan,d.nsub=d.dyn.shape
    d.freq=round(np.mean(d.freqs),2)
    d.df=round(float(np.median(np.diff(d.freqs))),6);d.bw=round(d.nchan*d.df,3)
    d.dt=float(np.mean(np.diff(d.times)));d.tobs=float(np.max(d.times)+d.dt)
    if np.std(np.diff(d.times))!=0:
        while len(d.times)>3:
            diffs=np.abs(np.diff(d.times));dt0=diffs[0];mean=np.mean(diffs[1:]);std=np.std(diffs[1:])
            if dt0-mean>=-2.33*std:break
            d.dyn=np.delete(d.dyn,0,axis=1);d.times=np.delete(d.times,0)
        d.mjd+=np.min(d.times)/86400.;d.times-=np.min(d.times)
        d.nsub=len(d.times);d.dt=round(np.mean(np.diff(d.times)),3);d.tobs=round(max(d.times)+d.dt,3)
    return d

def load_observations(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as data:
        return [make_spectrum(data[f'intensity_{i}'][0],data[f'frequency_{i}'][0],data[f'time_{i}'][0],data[f'mjd_{i}'][0,0]) for i in range(8)]

def stitch(first,second):
    """Exact source concatenation/gap-clock convention; no curvature calibration."""
    gap=round((second.mjd-first.mjd)*86400-first.tobs,1)
    n_extra=0 if gap<first.dt else len(np.arange(0,gap,first.dt))
    combined=Spectrum();combined.dyn=np.concatenate([first.dyn,np.zeros((first.nchan,n_extra)),second.dyn],axis=1)
    combined.nchan=first.nchan;combined.nsub=first.nsub+n_extra+second.nsub
    combined.tobs=first.tobs+gap+second.tobs;combined.times=np.linspace(0,combined.tobs,combined.nsub)
    for name in ['bw','df','freq','dt']:setattr(combined,name,getattr(first,name))
    combined.freqs=first.freqs.copy();combined.mjd=min(first.mjd,second.mjd)
    return combined
