"""Explicit source SVD, circulant SVD, oscillation truncation and leakage model.

Adapted from FAIR-Unipd DSC toolbox under MIT; see ../LICENSE. Source missing
oSVD inverse-time factor is corrected and documented, not claimed unchanged.
"""
import numpy as np
from scipy.linalg import svd,lstsq
from scipy.integrate import cumulative_trapezoid
from .physics_model import convolution_matrix

def oscillation_index(residue):
    """Source signed-peak second-difference index; negative lobes are retained."""
    r=np.asarray(residue)
    return np.abs(np.diff(r,n=2,axis=-1)).sum(axis=-1)/(r.shape[-1]*r.max(axis=-1))

def deconvolve(concentration,arterial_input,method='osvd',tr=1.55,oi_limit=.035,chunk_size=4096):
    """Deconvolve supported N-by80 observations into80/160 signed lag samples.

    Source fixed cutoffs: sSVD20%, cSVD10%; oSVD5:5:95%, first OI<.035.
    All160 lags participate in OI. oSVD peak is signed maximum; other peaks
    are maximum absolute values, preserving the original distinction.
    """
    y=np.asarray(concentration,dtype=np.float64);a=np.asarray(arterial_input)
    if method not in ('svd','csvd','osvd') or y.ndim!=2 or y.shape[1]!=len(a):raise ValueError('Invalid method or N-by-time input')
    if not np.isfinite(y).all() or tr<=0 or chunk_size<1:raise ValueError('Finite observations, positive TR/chunk required')
    matrix=convolution_matrix(a,method!='svd');u,s,vh=svd(matrix,full_matrices=False,lapack_driver='gesvd')
    out=np.empty((len(y),len(s)));indices=np.zeros(len(y));cutoffs=np.zeros(len(y))
    thresholds=np.arange(5,100,5)/100 if method=='osvd' else np.array([.2 if method=='svd' else .1])
    for start in range(0,len(y),chunk_size):
        stop=min(start+chunk_size,len(y));padded=np.zeros((stop-start,len(s)));padded[:,:len(a)]=y[start:stop]
        projected=padded@u;pending=np.ones(stop-start,dtype=bool)
        for threshold in thresholds:
            inv=np.zeros_like(s);keep=s>=threshold*s[0];inv[keep]=1/s[keep]
            r=((projected[pending]*inv)@vh)/tr
            positions=np.flatnonzero(pending);out[start+positions]=r
            cutoffs[start+positions]=threshold
            if method!='osvd':break
            oi=oscillation_index(r);indices[start+positions]=oi
            pending[positions[oi<oi_limit]]=False
            if not pending.any():break
    peak=out.max(axis=-1) if method=='osvd' else np.abs(out).max(axis=-1)
    return dict(residual=out,map=peak,oscillation=indices,cutoff=cutoffs)

def leakage_volume(concentration,arterial_input,support,bolus,tr=1.55):
    """Return relative CBV and source conditional-CV leakage regression stages.

    Full22 slices participate in baseline-count mean/min, including source
    empty-slice baseline count2. Tail median uses last11samples,2-SD criterion.
    These source conventions differ from prose comments and are preserved.
    """
    concentration=np.asarray(concentration,dtype=np.float64);support=np.asarray(support)
    y=concentration[support];time=np.arange(y.shape[1])*tr
    sd=np.std(y[:,:int(np.floor(np.mean(bolus)))],axis=1,ddof=1)
    tail=np.median(y[:,-11:],axis=1);nonenhancing=np.abs(tail)<=2*sd
    if not nonenhancing.any():raise ValueError('No nonenhancing voxels for specified leakage model')
    average=y[nonenhancing].mean(axis=0)
    integral=cumulative_trapezoid(average,time,initial=0)
    design=np.column_stack([-integral,average])
    parameters=lstsq(design,y.T,lapack_driver='gelsd')[0].T
    sigma=np.var(y[:,:int(np.min(bolus))],axis=1,ddof=1)
    covariance=np.linalg.inv(design.T@design)
    cv=100*np.sqrt(sigma[:,None]*np.diag(covariance)[None,:])/np.abs(parameters)
    denominator=np.trapezoid(arterial_input,time)
    cbv=np.trapezoid(y,time,axis=-1)/denominator
    corrected=cbv+(cv[:,0]<100)*parameters[:,0]*np.trapezoid(integral,time)/denominator
    fields={}
    for key,values in dict(cbv=cbv,cbv_lc=corrected,K1=parameters[:,1],K2=parameters[:,0],K1cv=cv[:,1],K2cv=cv[:,0]).items():
        result=np.zeros(support.shape);result[support]=values;fields[key]=result
    fields['nonenhancing']=np.zeros(support.shape,dtype=bool);fields['nonenhancing'][support]=nonenhancing
    fields['mean_non_enhancing']=average
    return fields
