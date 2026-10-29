"""Native distance-weighted time-domain full/deviatoric moment-tensor inversion."""
import numpy as np
from .preprocessing import DEPTHS,window_waveforms,distance_weights
from .physics_model import design_matrix,coefficients_to_tensor

def variance_reduction(observed,predicted,weights):
    """100*(1-weighted residual energy/weighted observed energy)."""
    d=np.asarray(observed);r=d-np.asarray(predicted)
    w=np.broadcast_to(np.asarray(weights)[:,None,None],d.shape)
    return float(100*(1-np.sum(w*r*r)/np.sum(w*d*d)))

def solve(data,degree=6,weight='distance',stations=None):
    """Fit all native depths; return tensors[Nm], depth[km], VR[%], predictions[cm].

    stations optionally selects a station subset for sensitivity experiments.
    Runtime defaults always use every native station and every depth.
    """
    d=window_waveforms(data['waveforms'])
    indices=np.arange(len(d)) if stations is None else np.asarray(stations)
    d=d[indices]
    w=distance_weights(data['station_distance'][indices]) if weight=='distance' else np.ones(len(indices))
    wd=np.repeat(w,3*d.shape[-1])
    tensors=[];predictions=[];vrs=[];condition=[]
    for gf in data['fundamental_greens'][:,indices]:
        G=design_matrix(gf,data['station_azimuth'][indices],degree)
        normal=G.T@(wd[:,None]*G)
        rhs=G.T@(wd*d.ravel())
        a=np.linalg.solve(normal,rhs)
        pred=(G@a).reshape(d.shape)
        tensors.append(coefficients_to_tensor(a));predictions.append(pred)
        vrs.append(variance_reduction(d,pred,w));condition.append(np.linalg.cond(normal))
    vrs=np.asarray(vrs);best=int(np.argmax(vrs))
    return {'reconstruction':np.array(tensors)[best][None], 'source_depth':DEPTHS[best:best+1].reshape(1,1),
            'tensors':np.array(tensors),'depths':DEPTHS.copy(),'variance_reduction':vrs,
            'predictions':np.array(predictions),'condition':np.array(condition),'best':best}
