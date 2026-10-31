"""Local RF-to-vector inverse and explicit robust weighted DCT-PLS updates.

Derived from official MUST/smoothn, Damien Garcia, LGPL-3.0-or-later.
"""
import numpy as np
from scipy.fft import dctn,idctn
from .physics_model import demodulate,beamform,doppler_velocity,invert_receive_projections,RECEIVE_ANGLE,SAMPLING_FREQUENCY

def laplacian_eigenvalues(shape):
    ny,nx=shape
    return (2-2*np.cos(np.pi*np.arange(ny)/ny))[:,None]+(2-2*np.cos(np.pi*np.arange(nx)/nx))[None,:]

def dct_filter(data,smoothing):
    gamma=1/(1+smoothing*laplacian_eigenvalues(data.shape)**2)
    return idctn(gamma*dctn(data,type=2,norm='ortho'),type=2,norm='ortho')

def robust_smooth(vector,mask,smoothing=1e4,robust=True,max_iter=100,tolerance=1e-3):
    """Three source robust steps, each with explicit relaxed weighted updates.

    Missing points have zero weight. The source's toolbox-free initial guess
    fills them by each component's observed mean then keeps its lowest ten
    percent of DCT frequencies per axis. No target or reference is read.
    """
    vector=np.asarray(vector,dtype=float);mask=np.asarray(mask,dtype=bool)
    if vector.ndim!=3 or vector.shape[0]!=2 or vector.shape[1:]!=mask.shape or not mask.any():raise ValueError('Expected two-component field and nonempty public mask')
    if not np.isfinite(vector[:,mask]).all() or smoothing<=0:raise ValueError('Invalid observed velocity or smoothing')
    data=np.where(mask[None],vector,0.)
    weighted=not mask.all();relaxation=1+.75*weighted
    if weighted:
        filled=np.where(mask[None],data,np.mean(data[:,mask],axis=1)[:,None,None])
        coefficients=dctn(filled,axes=(1,2),type=2,norm='ortho')
        ky,kx=np.ceil(np.asarray(mask.shape)/10).astype(int)
        coefficients[:,ky:,:]=0;coefficients[:,:,kx:]=0
        current=idctn(coefficients,axes=(1,2),type=2,norm='ortho')
    else:current=np.zeros_like(data)
    gamma=1/(1+smoothing*laplacian_eigenvalues(mask.shape)**2)
    weights=mask.astype(float);history=[]
    root=np.sqrt(1+16*smoothing)
    leverage=(np.sqrt(1+root)/np.sqrt(2)/root)**2
    for outer in range(3 if robust else 1):
        tol=1.;iteration=0;weights_used=weights.copy()
        while tol>tolerance and iteration<max_iter:
            previous=current.copy()
            rhs=weights[None]*(data-current)+current
            updated=idctn(gamma[None]*dctn(rhs,axes=(1,2),type=2,norm='ortho'),axes=(1,2),type=2,norm='ortho')
            current=relaxation*updated+(1-relaxation)*current
            denominator=np.linalg.norm(current)
            tol=float(weighted*np.linalg.norm(previous-current)/denominator) if denominator else 0.
            iteration+=1
        history.append({'robust_step':outer+1,'iterations':iteration,'relative_change':tol,'converged':iteration<max_iter})
        if robust:
            residual=data-current
            finite=residual[:,mask]
            median=np.median(finite,axis=1)
            mad=np.median(np.linalg.norm(finite-median[:,None],axis=0))
            if mad<=0:raise ArithmeticError('Undefined robust vector residual scale')
            student=np.linalg.norm(residual,axis=0)/(1.4826*mad)/np.sqrt(1-leverage)
            ratio=student/4.685
            weights=mask*(1-ratio**2)**2*(ratio<1)
            weighted=True
    return current,{'history':history,'weights_used':weights_used,'weights_after_final_step':weights,'leverage':leverage,'smoothing':smoothing}

def reconstruct(observations,window=11,lag=1,smoothing=1e4,robust=True,receive_angle=RECEIVE_ANGLE,sampling_frequency=SAMPLING_FREQUENCY):
    """Recover both complete measured-disk components from all32 RF pulses."""
    iq=demodulate(observations['rf'],sampling_frequency=sampling_frequency)
    x,z=np.meshgrid(observations['x'],observations['z'])
    beams=[];radial=[];variance=[];ac=[]
    for angle in [receive_angle,-receive_angle]:
        beam=beamform(iq,x,z,angle,sampling_frequency=sampling_frequency)
        v,var,a=doppler_velocity(beam,window=window,lag=lag)
        beams.append(beam);radial.append(v);variance.append(var);ac.append(a)
    raw=invert_receive_projections(*radial,receive_angle=receive_angle)
    smooth,diagnostics=robust_smooth(raw,observations['roi'],smoothing=smoothing,robust=robust)
    output=smooth.copy();output[:,~observations['roi']]=0
    return {'velocity':output,'smoothed_full':smooth,'raw_velocity':raw,'iq':iq,'beam_iq':beams,
            'radial':np.stack(radial),'variance':np.stack(variance),'autocorrelation':np.stack(ac),
            'smoothing':diagnostics,'x':x,'z':z,'roi':observations['roi']}
