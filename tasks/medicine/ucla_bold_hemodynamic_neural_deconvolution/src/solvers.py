"""Complete canonical HRF retrieval and relative neural-drive inverse."""
import numpy as np
from .preprocessing import preprocess
from .physics_model import parameters,canonical_basis_raw,orthogonalize_basis,BOLD_event_vector,hrf_fit,deconvolve_ridge
def reconstruct(data,derivatives=2):
    """Fit all400anatomical parcels/all152frames; return outputs+diagnostics."""
    if derivatives not in [0,1,2]:raise ValueError('derivatives must be0,1,2')
    _,filtered=preprocess(data['bold']);para=parameters();para['TD_DD']=derivatives
    bf=orthogonalize_basis(canonical_basis_raw(para))
    n,nroi=filtered.shape;coeff=np.zeros((bf.shape[1]+2,nroi));events=np.zeros((nroi,n),dtype=np.uint8)
    for i in range(nroi):
        peak=BOLD_event_vector(n,filtered[:,i],np.atleast_1d(para['thr']),para['localK'],[])
        events[i]=peak.toarray()[0]
        u=np.append(peak.toarray(),np.zeros((para['T']-1,n)),axis=0).reshape((1,-1),order='F')
        coeff[:,i]=hrf_fit(filtered[:,i],para,u,bf)
    hrf=np.dot(bf,coeff[:bf.shape[1]]);drive=deconvolve_ridge(filtered,hrf)
    return {'hrf':hrf.T[None],'neural_drive':drive.T[None],'coefficients':coeff,'events':events,'filtered_bold':filtered,'basis':bf}
def submission(result):
    return {k:result[k] for k in ['hrf','neural_drive']}
