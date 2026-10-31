"""Local calibrated Tikhonov singular-system reconstruction.

The published eta/cutoff/complex64 Fourier coefficient convention is retained.
Only odd-grid negative-frequency coverage is qualified, as independently proved.
"""
import numpy as np
from .preprocessing import prepare_observations
from .physics_model import transfer_fields,frequency_matrix,forward_linear

def tikhonov_inverse(g,h,eta=.7):
    """Return XYZ15 float64 coefficients from XYZPV real measurements.

    At each spatial frequency, for H(J,D)=U diag(s) Vh, evaluate
    F=U diag(s/(s²+eta)) Vh G with singular values<=1e-7 suppressed.
    Eta is dimensionless in the source's normalized OTF convention.
    """
    if not np.isfinite(eta) or eta<0:raise ValueError('eta must be finite and nonnegative')
    shape=g.shape[:3];X,Y,Z=shape
    # Original NumPy1.26 forward transform promotes real float32 input.
    G=np.fft.rfftn(np.asarray(g,dtype=np.float64),axes=(0,1,2)).reshape(X,Y,Z//2+1,-1)
    F=np.empty((X,Y,Z//2+1,15),np.complex64)
    ax=np.abs(np.rint(np.fft.fftfreq(X)*X)).astype(int)
    ay=np.abs(np.rint(np.fft.fftfreq(Y)*Y)).astype(int)
    for z in range(Z//2+1):
        H=frequency_matrix(h,z,shape,positive_only=True)
        u,s,vh=np.linalg.svd(H,full_matrices=False)
        sreg=np.where(s>1e-7,s/(s*s+eta),0)
        pinv=np.einsum('xysv,xyv,xyvd->xysd',u,sreg,vh,optimize=True)
        # Absolute FFT-frequency pairing is exact for both odd and even sizes.
        F[:,:,z]=np.einsum('xyjd,xyd->xyj',pinv[ax[:,None],ay[None,:]],G[:,:,z])
    return np.fft.irfftn(F.astype(np.complex128),s=shape,axes=(0,1,2))

def reconstruct(raw,subset='ALL',eta=None,calibrate=True):
    """Actively solve one published group; return field plus named diagnostics.

    Input raw mapping contains batch-first uint16 real observations and lake.
    No references or fixture files are accessed. Calibration uses independent
    lake counts and an isotropic unit-source prediction, never target fitting.
    """
    p=prepare_observations(raw,subset)
    if eta is None:eta=.1 if subset=='SIX-NO-TILT' else .7
    h=transfer_fields(p['pols'],p['observations'].shape[:3])
    unit=np.zeros(p['observations'].shape[:3]+(15,));unit[...,0]=1
    response=forward_linear(unit,h);response/=np.max(response)
    model_averages=response.mean(axis=(0,1,2));del unit,response
    g=p['observations']
    if calibrate:
        c=p['calibration_averages']
        g=g*c[0,0]*model_averages/c.T
    field=tikhonov_inverse(g,h,eta)
    prediction=forward_linear(field,h)
    relative=float(np.linalg.norm(prediction-g)/np.linalg.norm(g))
    return {'reconstruction':field,'prediction':prediction,'corrected_observations':g,'transfer_fields':h,'relative_forward_residual':relative,'model_averages':model_averages,'calibration':p,'eta':float(eta),'subset':subset,'calibrated':bool(calibrate)}

def isotropic_null_residuals(g,h,eta=.7):
    """Complete-domain input-only nulls; these are scientific comparisons only.

    Fit one spatially constant isotropic amplitude, or an independent scalar
    f00 at each spatial Fourier bin, with/without the specified ridge penalty.
    All angular coefficients1..14 are fixed zero in these null models. The
    unregularized null uses the source singular cutoff, with no output clipping.
    """
    shape=g.shape[:3];X,Y,Z=shape
    G=np.fft.rfftn(np.asarray(g,dtype=float),axes=(0,1,2)).reshape(X,Y,Z//2+1,-1)
    power=0.;unregularized=0.;regularized=0.
    for z in range(Z//2+1):
        a=frequency_matrix(h,z,shape)[:,:,0,:]
        denominator=np.sum(a*a,axis=-1);numerator=np.sum(a*G[:,:,z],axis=-1)
        scalar=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-14)
        pred=a*scalar[...,None];ridge=a*(numerator/(denominator+eta))[...,None]
        weight=1 if z==0 or (Z%2==0 and z==Z//2) else 2
        power+=weight*np.sum(abs(G[:,:,z])**2)
        unregularized+=weight*np.sum(abs(pred-G[:,:,z])**2)
        regularized+=weight*np.sum(abs(ridge-G[:,:,z])**2)
    a=frequency_matrix(h,0,shape)[0,0,0].reshape(g.shape[-2:])
    amplitude=float(np.sum(g*a)/(np.prod(shape)*np.sum(a*a)))
    return {'zero':1.,'uniform_isotropic':float(np.linalg.norm(g-amplitude*a)/np.linalg.norm(g)),'spatial_isotropic_unregularized':float(np.sqrt(unregularized/power)),'spatial_isotropic_regularized':float(np.sqrt(regularized/power)),'uniform_amplitude':amplitude}
