"""Explicit conservative native-Q overlap and measured-reference reduction."""
from pathlib import Path
import numpy as np

CONFIG={'360':(.6,1.8,8.,.5,12,92),'197':(.55,1.58,9.,1.,14,169)}

def load_raw(path):
    with np.load(path,allow_pickle=False) as f: raw={k:f[k] for k in f.files}
    for name,(_,_,_,_,nq,ne) in CONFIG.items():
        shapes={'signal':(1,3,249,320),'standard_error':(1,3,249,320),'q_edges':(1,250),'energy_edges':(1,321),'analysis_q_edges':(1,31 if name=='360' else 30),'analysis_energy_index':(1,ne),'acquisition_mask':(1,3,249,320)}
        for key,shape in shapes.items():
            a=raw[key+'_'+name]
            dtype=np.dtype(bool if key=='acquisition_mask' else np.int64 if key=='analysis_energy_index' else np.float64)
            if a.shape!=shape or a.dtype!=dtype: raise ValueError('Invalid raw '+key+'_'+name)
            if key not in ('signal','standard_error') and not np.isfinite(a).all(): raise ValueError('Nonfinite calibration')
        mask=raw['acquisition_mask_'+name]
        for key in ('signal','standard_error'):
            a=raw[key+'_'+name]
            if np.isinf(a).any() or not np.array_equal(np.isfinite(a),mask):raise ValueError('Native acquisition-mask mismatch')
        if np.any(raw['standard_error_'+name][mask]<0):raise ValueError('Negative standard error')
    if len(raw)!=14:raise ValueError('Unexpected raw keys')
    return raw

def overlap_rebin(signal,standard_error,old_edges,new_edges,q_limits,divisor):
    overlap=np.maximum(0.,np.minimum(new_edges[1:,None],old_edges[None,1:])-np.maximum(new_edges[:-1,None],old_edges[None,:-1]))
    weights=overlap/np.diff(old_edges)[None,:]
    mid=(old_edges[:-1]+old_edges[1:])/2
    weights[:,(mid<q_limits[0])|(mid>q_limits[1])]=0
    values=[];variances=[]
    for w in weights:
        active=w>0
        values.append(np.sum(signal[:,active,:]*w[active][None,:,None],axis=1)/divisor)
        variances.append(np.sum(standard_error[:,active,:]**2*w[active][None,:,None]**2,axis=1)/divisor**2)
    return np.stack(values,axis=1),np.stack(variances,axis=1)

def prepare(raw):
    data={}
    for name,(lo,hi,divisor,scale,nq,ne) in CONFIG.items():
        edges=raw['analysis_q_edges_'+name][0];q=(edges[:-1]+edges[1:])/2
        qi=np.flatnonzero((q>=lo)&(q<=hi));ei=raw['analysis_energy_index_'+name][0]
        values,variances=overlap_rebin(raw['signal_'+name][0],raw['standard_error_'+name][0],raw['q_edges_'+name][0],edges,(lo,hi),divisor)
        v=values[:,qi,:][:,:,ei];var=variances[:,qi,:][:,:,ei]
        y=(v[0]-v[2])*scale;sigma=np.sqrt(var[0]+var[2])*scale;response=v[1]-v[2]
        e=raw['energy_edges_'+name][0];energy=((e[:-1]+e[1:])/2)[ei]
        if y.shape!=(nq,ne) or not all(np.isfinite(a).all() for a in (y,sigma,response)) or np.any(sigma<=0) or np.any(response.sum(axis=1)<=0):raise ValueError('Invalid selected observations')
        data[name]=dict(q=q[qi],energy=energy,y=y,sigma=sigma,response=response,resolution=response/response.sum(axis=1)[:,None],rebinned_signal=v,rebinned_variance=var,q_index=qi,energy_index=ei)
    return data
