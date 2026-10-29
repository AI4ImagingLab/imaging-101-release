"""Original Fernald direct inverse followed by source in-place smoothing."""
import numpy as np
from .physics_model import reference_integral,START,REFERENCE,OUTPUT,MOLECULAR_LIDAR_RATIO,AEROSOL_LIDAR_RATIO

def source_smooth(values,width=11):
    """Preserve source aliased input/output sequential moving mean."""
    out=values.copy();half=width//2
    for i in range(out.shape[-1]):
        # The source accumulates neighbors left to right; earlier cells have
        # already been replaced. This is not a symmetric convolution.
        total=np.zeros(out.shape[0]);count=0
        for j in range(max(0,i-half),min(out.shape[-1],i+half+1)):
            total+=out[:,j];count+=1
        out[:,i]=total/count
    return out

def fernald(signal,molecular_backscatter,molecular_extinction,lidar_ratio=AEROSOL_LIDAR_RATIO,smooth=True):
    int_mol=reference_integral(molecular_extinction)
    ka=1/abs(lidar_ratio);km=1/MOLECULAR_LIDAR_RATIO
    normalized=signal/signal[:,REFERENCE,None]
    p=normalized*np.exp(-2*((km-ka)/ka)*int_mol)
    p[:,:START]=0
    integral=reference_integral(p)
    total=p/(1/molecular_backscatter[REFERENCE]-(2/ka)*integral)
    beta=total-molecular_backscatter
    beta[:,:START+1]=beta[:,START+1,None]
    alpha=beta*lidar_ratio
    raw_beta=beta.copy();raw_alpha=alpha.copy()
    if smooth:beta=source_smooth(beta);alpha=source_smooth(alpha)
    return beta,alpha,{'p':p,'integral_p':integral,'integral_molecular':int_mol,'normalized_signal':normalized,'beta_unsmoothed':raw_beta,'alpha_unsmoothed':raw_alpha}

def reconstruct(data,lidar_ratio=AEROSOL_LIDAR_RATIO,smooth=True):
    beta,alpha,stages=fernald(data['range_corrected_signal'],data['molecular_backscatter'],data['molecular_extinction'],lidar_ratio,smooth)
    return {'reconstruction':beta[:,OUTPUT][None],'extinction':alpha[:,OUTPUT][None]},stages
