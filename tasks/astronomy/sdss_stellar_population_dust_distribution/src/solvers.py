"""Explicit BIC branching and complete mixture exploration, adapted from FIREFLY.
MIT copyright 2026 FIREFLY Collaboration; retain task LICENSE.
"""
import numpy as np
from .preprocessing import normalize_models
from .physics_model import highpass,calzetti,smooth_curve,forward
def sigma_clip(values,sigma=3.):
    selected=np.asarray(values).ravel().copy()
    while True:
        lower=np.mean(selected)-sigma*np.std(selected);upper=np.mean(selected)+sigma*np.std(selected)
        reduced=selected[(selected>=lower)&(selected<=upper)]
        if len(reduced)==len(selected):return reduced,lower,upper
        if not len(reduced):raise ValueError('Empty sigma-clipped sample')
        selected=reduced
def fit_population(flux,error,models,max_iterations=10):
    m,n=models.shape;chi_models=(models-flux)/error;weights=[];chis=[];branches=[];masks=[];mask=None
    def score(w,branch):
        nonlocal mask
        ix=np.nonzero(w);r=np.dot(w[ix],chi_models[ix])
        if branch==0:
            clipped,lower,upper=sigma_clip(r);mask=(r>lower)&(r<upper);return float(np.sum(clipped**2))
        return float(np.sum(r[mask]**2))
    for j in range(m):
        w=np.zeros(m);w[j]=1;weights.append(w);chis.append(score(w,0));branches.append(0);masks.append(mask.copy())
    mask=masks[np.argmin(chis)].copy();initial_mask=mask.copy();history=[]
    for iteration in range(1,max_iterations+1):
        count=len(weights);threshold=min(chis)-np.log(n)
        for j in range(count):
            new=weights[j]*iteration;denominator=np.sum(new)+1
            for k in range(m):
                new[k]+=1;w=new/denominator;c=score(w,iteration);new[k]-=1
                if c<threshold:weights.append(w);chis.append(c);branches.append(iteration)
        history.append([iteration,len(weights),min(chis)])
        if len(weights)==count:break
    else:raise RuntimeError('BIC branching did not converge in selected ten iterations')
    retained=len(weights);best=np.argsort(chis)[:min(10,retained)]
    for first in best:
        for second in range(retained):
            for q in [1e-7,1e-6,1e-5,1e-4,.001,.01,.1,1.]:
                w=(weights[first]+q*weights[second])/(1+q);branch=branches[first]+branches[second]
                weights.append(w);chis.append(score(w,branch));branches.append(branch)
    chis=np.array(chis);weights=np.array(weights);best=int(np.argmin(chis))
    return dict(weights=weights,chis=chis,branches=np.array(branches),best_weights=weights[best],best_index=best,mask=initial_mask,history=np.array(history),retained=retained)
def determine_attenuation(data,normalized):
    flux=data['flux'];wave=data['wavelength'];hflux=highpass(flux)
    hmodels=normalize_models(hflux,np.array([highpass(x) for x in normalized]))
    herror=np.full(len(flux),np.median(data['uncertainty'])/np.median(flux)*np.median(hflux))
    fit=fit_population(hflux,herror,hmodels);w=fit['best_weights'];best=w@normalized;best_hpf=w@hmodels
    with np.errstate(divide='ignore',invalid='ignore'):fine=flux/best-hflux/best_hpf+1
    bad=~np.isfinite(fine);fine[bad]=1
    herror[bad]=np.max(herror)*9999999999.9
    fine/=np.median(fine)
    attenuation=smooth_curve(wave,fine)
    grid=1.5*np.arange(200)/199;loss=[]
    for e in grid:
        law=calzetti(e,wave);law/=np.median(law);clipped,_,_=sigma_clip((attenuation-law)**2);loss.append(np.sum(clipped**2))
    return attenuation,float(grid[np.argmin(loss)]),dict(fit=fit,flux=hflux,error=herror,templates=hmodels,fine=fine,bad=bad,dust_grid=grid,dust_loss=np.array(loss))
def reconstruct(data):
    normalized=normalize_models(data['flux'],data['templates'])
    attenuation,ebv,high=determine_attenuation(data,normalized)
    models=normalize_models(data['flux'],data['templates']*attenuation)
    final=fit_population(data['flux'],data['uncertainty'],models)
    output=dict(reconstruction=final['best_weights'][None],attenuation=attenuation[None],ebv=np.array([[ebv]]))
    return output,dict(highpass=high,final=final,prediction=forward(final['best_weights'],attenuation,data['templates'],data['flux']),templates=models)
