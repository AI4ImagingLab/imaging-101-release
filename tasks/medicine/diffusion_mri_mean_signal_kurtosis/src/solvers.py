"""Local published MSDKI weighted log-linear inverse, DIPY BSD-3-Clause."""
import numpy as np
from .preprocessing import shell_average,signal_mask,MIN_SIGNAL
from .physics_model import design_matrix

def fit_shells(means,counts,bvalues,mask,weighted=True):
    """Return D,K,S0 using original normal-equation pseudoinverse.

    Weights n_direction*observed_shell_mean**2. Closed-form inverse requires
    no initialization, iterative optimizer, line search or stopping criterion.
    Negative K is preserved as evidence of measurement/model limitations.
    """
    y=np.maximum(np.asarray(means,dtype=np.float64),MIN_SIGNAL)
    B=design_matrix(bvalues);out=np.zeros(y.shape[:-1]+(3,))
    for v in np.ndindex(mask.shape):
        if not mask[v] or np.mean(y[v])<=MIN_SIGNAL:continue
        W=np.diag(counts*y[v]**2 if weighted else np.ones_like(counts));BTW=B.T@W
        p=np.linalg.multi_dot([np.linalg.pinv(BTW@B),BTW,np.log(y[v])])
        p[1]/=p[0]**2;p[2]=np.exp(p[2]);out[v]=p
    return out[...,0],out[...,1],out[...,2]

def reconstruct(data,weighted=True,shell_selection=None):
    """Return three full native-slice maps, float64 (1,3,96,96)."""
    signals=data['signals'][0];b=data['bvalues'][0];mask=signal_mask(signals,b)
    if shell_selection is not None:
        keep=np.isin(b,shell_selection);signals=signals[...,keep];b=b[keep]
    means,counts,shells=shell_average(signals,b)
    D,K,S0=fit_shells(means,counts,shells,mask,weighted)
    return {'diffusivity':D[None],'reconstruction':K[None],'baseline':S0[None]}
