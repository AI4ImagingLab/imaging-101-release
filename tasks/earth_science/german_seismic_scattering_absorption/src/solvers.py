"""Profile weighted inversion and golden-section scattering optimization.
Adapted from Qopen core.py, copyright Tom Eulenfeld, MIT; see LICENSE.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from .preprocessing import prepare_band
from .physics_model import green_observations
SCATTERING_BOUNDS=(1e-8,1e-4);ABSORPTION_BOUNDS=(1e-3,10.)
def design_system(pairs):
    ns=len(pairs);parts=[]
    for j,p in enumerate(pairs):
        a=np.zeros((len(p['tcoda']),ns+1));a[:,j]=1;a[:,-1]=-p['tcoda'];parts.append(a)
    bulk=np.zeros((ns,ns+1));bulk[np.arange(ns),np.arange(ns)]=1;bulk[:,-1]=-np.array([p['tbulk'] for p in pairs]);parts.append(bulk)
    design=np.vstack(parts);energy=np.r_[np.concatenate([p['Ecoda'] for p in pairs]),[p['Ebulk'] for p in pairs]]
    weights=np.r_[np.ones(len(energy)-ns),[p['bulk_weight'] for p in pairs]]
    return design,energy,weights
def weighted_solve(design,target,weights):
    x=design*np.sqrt(weights)[:,None];y=target*np.sqrt(weights)
    u,s,vt=np.linalg.svd(x,full_matrices=False);reciprocal=np.where(s>1e-15*s.max(),1/s,0.)
    inverse=np.dot(vt.T,np.multiply(reciprocal[:,None],u.T));coeff=np.dot(inverse,y)
    residual=y-x@coeff;rank=np.linalg.matrix_rank(np.diag(s))
    return coeff,float(np.sqrt(np.dot(residual,residual)/(len(y)-rank)))
def solve_scattering(pairs,scattering,system=None):
    design,energy,weights=design_system(pairs) if system is None else system
    coda,bulk=green_observations(pairs,scattering);g=np.r_[np.concatenate(coda),bulk]
    coeff,error=weighted_solve(design,np.log(energy)-np.log(g),weights)
    source=np.exp(np.mean(coeff[:-1]));sites=np.exp(coeff[:-1])/source
    return dict(scattering=scattering,absorption=coeff[-1],source=source,sites=sites,error=error,Gcoda=coda,Gbulk=bulk,coefficients=coeff,model=g*np.exp(design@coeff))
def reconstruct_band(pairs):
    system=design_system(pairs)
    def objective(g):
        if not SCATTERING_BOUNDS[0]<=g<=SCATTERING_BOUNDS[1]:return np.inf
        r=solve_scattering(pairs,g,system)
        return r['error'] if ABSORPTION_BOUNDS[0]<r['absorption']<ABSORPTION_BOUNDS[1] else np.inf
    fit=minimize_scalar(objective,method='golden',bracket=SCATTERING_BOUNDS,tol=1e-3)
    if not fit.success:raise RuntimeError(fit.message)
    return solve_scattering(pairs,fit.x,system)
def reconstruct(data):
    parameters=np.zeros((5,5,2));source=np.zeros((5,5));sites=np.zeros((5,5,5));details=[]
    for e in range(5):
        row=[]
        for f in range(5):
            pairs=prepare_band(data,e,f);r=reconstruct_band(pairs);parameters[e,f]=[r['scattering'],r['absorption']];source[e,f]=r['source'];sites[e,f,[p['station_slot'] for p in pairs]]=r['sites'];r['pairs']=pairs;row.append(r)
        details.append(row)
    return dict(reconstruction=parameters[None],source_energy=source[None],site_amplification=sites[None]),details
