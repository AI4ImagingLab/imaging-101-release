"""Explicit approximate reweighted augmented-Lagrangian mathematics.
Adapted from BaSiCPy2.0, peng-lab/BaSiCPy; MIT notice at task LICENSE.
"""
import numpy as np
import json
from scipy.fft import dctn,idctn
from .preprocessing import resize_stack,upsample_field
def shrink(x,t):return np.sign(x)*np.maximum(abs(x)-t,0)
def dct(x):return dctn(x,type=2,norm='ortho',axes=(-2,-1))
def idct(x):return idctn(x,type=2,norm='ortho',axes=(-2,-1))

def step(Im,W,Shat,R,B,Y,mu,max_mu,smoothness):
    S=idct(Shat)
    IB=B[:,None,None]*S
    delta=np.mean(Im-R-IB+Y/mu,axis=0)
    Shat=shrink(Shat+dct(delta),np.float32(smoothness/mu))
    S=idct(Shat)
    IB=B[:,None,None]*S
    R=shrink(R+(Im-IB-R+Y/mu),W/mu)
    background=Im-R
    B=np.maximum(background.mean(axis=(1,2))/background.mean(),0)
    IB=B[:,None,None]*S
    residual=Im-IB-R
    Y=Y+mu*residual
    mu=np.minimum(mu*np.float32(1.5),max_mu)
    return S,Shat,IB,R,B,Y,mu,residual

def fit_coarse(Im,smoothness=5.,max_iterations=500,max_reweight_iterations=10):
    Im=np.asarray(Im,dtype=np.float32)
    spectral=np.linalg.svd(Im.reshape(len(Im),-1),compute_uv=False)[0]
    init_mu=np.float32(12.5/spectral);max_mu=np.float32(init_mu*1e7)
    image_norm=np.linalg.norm(Im.ravel())
    W=np.ones_like(Im);last=None;history=[];first=None
    for outer in range(max_reweight_iterations):
        S=np.ones(Im.shape[-2:],dtype=np.float32);Shat=dct(S)
        B=Im.mean(axis=(1,2))/Im.mean()
        R=np.zeros_like(Im);Y=np.zeros_like(Im);mu=init_mu
        for k in range(max_iterations):
            S,Shat,IB,R,B,Y,mu,residual=step(Im,W,Shat,R,B,Y,mu,max_mu,smoothness)
            ratio=np.linalg.norm(residual.ravel())/image_norm
            if first is None:first=dict(S=S.copy(),Shat=Shat.copy(),IB=IB.copy(),R=R.copy(),B=B.copy(),Y=Y.copy(),mu=mu,residual=residual.copy())
            if ratio<=.001:break
        S=IB.mean(axis=0);meanS=S.mean();S=S/meanS
        weight=1/(abs(R/(IB.mean(axis=(1,2))[:,None,None]+np.float32(1e-6)))+np.float32(.1))
        W=weight*np.float32(weight.size)/weight.sum()
        change=None if last is None else float(np.sum(abs(S-last))/np.sum(abs(last)))
        history.append(dict(reweight=outer,iterations=k+1,residual_ratio=float(ratio),field_relative_change=change,converged=bool(k+1<max_iterations)))
        if change is not None and change<=.01:break
        last=S.copy()
    return dict(flatfield=S,baseline=B*meanS,residual=R,history=history,first=first,converged=not(outer==max_reweight_iterations-1 and k+1>=max_iterations),init_mu=float(init_mu))

def fit_baseline(Im,S):
    Im=np.asarray(Im,dtype=np.float32)
    threshold=np.quantile(Im.reshape(len(Im),-1),.1,axis=1).astype(np.float32)
    mask=Im<threshold[:,None,None]
    W=np.where(mask,np.float32(1),np.float32(.1))
    B_initial=np.nan_to_num(np.nanmean(np.where(mask,Im,np.nan),axis=(1,2)))
    spectral=np.linalg.svd(Im.reshape(len(Im),-1),compute_uv=False)[0]
    init_mu=np.float32(12.5/spectral);max_mu=np.float32(init_mu*1e7);norm=np.linalg.norm(Im.ravel())
    for outer in range(5):
        B=B_initial.copy();R=np.zeros_like(Im);Y=np.zeros_like(Im);mu=init_mu
        for k in range(500):
            IB=B[:,None,None]*S
            R=shrink(R+(Im-IB-R+Y/mu),W/mu)
            B=np.maximum(np.mean(Im-R,axis=(1,2)),0)
            residual=Im-IB-R
            Y=Y+mu*residual;mu=np.minimum(mu*np.float32(1.5),max_mu)
            if np.linalg.norm(residual.ravel())/norm<=.001:break
        weight=1/(np.abs(R/B[:,None,None])+np.float32(.1))
        weight[~mask]*=np.float32(.1)
        W=weight/weight.mean()*mask
    return B

def corrected_quantiles(images,field,baseline,stride,quantiles):
    pieces=[];size=images.shape[1]*images.shape[2]
    for i,frame in enumerate(images):
        corrected=frame/field-baseline[i]
        offset=(-i*size)%stride
        pieces.append(corrected.ravel()[offset::stride].copy())
    sample=np.concatenate(pieces)
    return np.quantile(sample,quantiles,overwrite_input=True).astype(np.float32)

def differential_entropy(images,field,baseline,vmin,vmax):
    edges=np.linspace(vmin,vmax,1001,dtype=np.float32);counts=np.zeros(1000,dtype=np.int64)
    for i,frame in enumerate(images):
        corrected=frame/field-baseline[i]
        counts+=np.histogram(corrected,bins=edges)[0]
    widths=np.diff(edges)
    density=counts.astype(np.float32)/np.float32(counts.sum())/widths
    positive=density>0
    return float(-np.sum(density[positive]*np.log(density[positive]))*widths[0])

def fourier_fraction(field):
    spectrum=abs(dctn(field,type=2,norm=None,axes=(-2,-1)))
    y,x=np.meshgrid(np.arange(128),np.arange(128),indexing='ij')
    outside=(x*x+y*y>100)&(x>0)&(y>0)
    return float(np.float32(np.sum(spectrum[outside]>.1))/np.float32(outside.sum()))

def autotune(images):
    pool=np.array([.01,.1,.5,.75,1,1.25,1.5,1.75,2,2.5,3,4,5,6,7,8,10])
    coarse=np.array([.01,.1,.5,2,8,10])
    selected=images[::max(len(images)//50,1)]
    ratio=selected.shape[1]*selected.shape[2]/(1024*1024)
    if ratio>1:
        reduced=resize_stack(selected,(int(selected.shape[1]/ratio),int(selected.shape[2]/ratio)))
    else:reduced=selected.astype(np.float32)
    small=resize_stack(reduced,(128,128))
    stride=max(1,int(np.ceil(reduced.size/2**24)))
    initial=fit_coarse(small,smoothness=float(pool.mean()))
    initial_field=upsample_field(initial['flatfield'],reduced.shape[-2:])
    baseline=fit_baseline(small,resize_stack(initial_field[None],(128,128))[0])
    low,high=corrected_quantiles(reduced,initial_field,baseline,stride,[.01,.99])
    value_range=(high-low*np.float32(.6))*np.float32(1.5)
    rows=[];fields=[];baseline_values=[]
    def evaluate(value):
        result=fit_coarse(small,smoothness=float(value))
        field=upsample_field(result['flatfield'],reduced.shape[-2:])
        b=fit_baseline(small,resize_stack(field[None],(128,128))[0])
        vmin=corrected_quantiles(reduced,field,b,stride,[.01])[0]*np.float32(.6)
        entropy=differential_entropy(reduced,field,b,vmin,vmin+value_range)
        frequency=fourier_fraction(result['flatfield'])
        score=np.float32(entropy)+np.float32(30)*np.float32(frequency)
        if np.allclose(field,1) or not result['converged']:score=np.inf
        row=dict(smoothness_flatfield=float(value),cost=float(score),entropy=float(entropy),fourier_fraction=frequency,entropy_vmin=float(vmin),entropy_vmax=float(vmin+value_range))
        rows.append(row);fields.append(result['flatfield'].copy());baseline_values.append(result['baseline'].copy())
        print('FRESH_COMPLETE_SEARCH',json.dumps(row),flush=True)
        return float(score)
    costs=[evaluate(value) for value in coarse]
    best=int(np.argmin(costs))
    if best==len(costs)-1:other=best-1
    elif best==0:other=1
    else:other=best+1;best-=1
    lower,upper=sorted([coarse[best],coarse[other]])
    narrow=pool[(pool>=lower)&(pool<=upper)]
    if len(narrow)==2:narrow=np.array([narrow[0],narrow.mean(),narrow[1]])
    narrow_costs=[costs[int(np.flatnonzero(coarse==narrow[0])[0])]]
    narrow_costs += [evaluate(value) for value in narrow[1:-1]]
    narrow_costs += [costs[int(np.flatnonzero(coarse==narrow[-1])[0])]]
    chosen=float(narrow[int(np.argmin(narrow_costs))])
    return dict(smoothness=chosen,search=rows,fields=np.asarray(fields),mainfit_baselines=np.asarray(baseline_values),initial_value_range=float(value_range),initial_quantiles=[float(low),float(high)])



def reconstruct(images):
    tuned=autotune(images)
    coarse=resize_stack(images,(128,128))
    result=fit_coarse(coarse,tuned["smoothness"])
    result["search"]=tuned["search"]
    result["smoothness"]=tuned["smoothness"]
    result["search_fields"]=tuned["fields"]
    result["search_mainfit_baselines"]=tuned["mainfit_baselines"]
    return result
