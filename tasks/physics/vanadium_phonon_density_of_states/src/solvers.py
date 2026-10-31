"""Local published multiphonon/MS correction and normalized DOS updates."""
import numpy as np
from .physics_model import DOS_E,spectral_orders,debye_waller,forward
from .preprocessing import preprocess
def initial_density():
    g=np.ones(80);g[DOS_E>55]=0
    small=DOS_E<55/3
    g[small]=(DOS_E[small]/(55/3))**2
    return g/g.sum()

def single_update(corrected,errors,q,prior):
    ee,A=spectral_orders(DOS_E,prior,orders=1)
    dw=debye_waller(q,DOS_E,prior)
    sim=(dw*np.exp(-dw))[:,None]*A[0,len(DOS_E)-1:][None]
    data=corrected[:,15:].copy();var=errors[:,15:].copy()
    bad=np.isnan(data);data[bad]=0;var[bad]=0;sim[bad]=0
    expse=data.sum(axis=0);variance=var.sum(axis=0);simse=sim.sum(axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        updated=prior*expse/simse
        relative_error=np.sqrt(variance)/expse
    updated[np.isnan(updated)]=0
    updated[:7]=DOS_E[:7]**2*updated[7]/49
    updated=np.maximum(updated,0)
    updated_variance=(updated*relative_error)**2
    out=prior.copy();out[:56]=updated[:56]
    v=np.zeros_like(prior);v[:56]=updated_variance[:56]
    norm=out.sum()
    return out/norm,v/norm**2

def reconstruct(intensity,variance,q,energy,max_iterations=20,tolerance=1e-4):
    measured,errors=preprocess(intensity,variance,q,energy)
    counts=measured.copy();errors=errors.copy();corrected=counts
    prior=initial_density();history=[];stages=[]
    for iteration in range(max_iterations):
        g,v=single_update(corrected,errors,q,prior)
        model=forward(g,q,np.isfinite(measured))
        factor=np.nansum(model['total'][:,22:])/np.nansum(counts[:,22:])
        counts*=factor;errors*=factor**2
        corrected=counts-model['mp']-model['ms']
        residual=corrected-model['sp']
        history.append(g.copy())
        stages.append(dict(**model,scaled_observation=counts.copy(),corrected=corrected.copy(),residual=residual.copy(),variance=v.copy()))
        if iteration>0 and np.allclose(g,prior,rtol=tolerance,atol=tolerance):break
        prior=g
    final_variance=v.copy()
    relative=abs(np.nansum(residual[:,22:71],axis=0))/np.nansum(counts[:,22:71],axis=0)
    final_variance[7:56]+=(g[7:56]*relative)**2
    return dict(reconstruction=g,variance=final_variance,history=np.asarray(history),stages=stages,processed_intensity=measured,processed_variance=preprocess(intensity,variance,q,energy)[1])

