"""Explicit two-half ensemble stretch sampler; no emcee/domain imports.
Stretch-move transition follows Goodman/Weare and emcee's MIT implementation.
"""
import numpy as np
from scipy.optimize import leastsq
from .physics_model import paired_lightcurves,eigencurves,design_matrix,spherical_coefficients,harmonics

def initialize_parameters(design,observed):
    def residual(p):
        model=p[0]*design[:,0]+p[1]
        for i in range(2,len(p)):model=model+p[i]*design[:,i]
        return observed-model
    return leastsq(residual,1e-4*np.ones(design.shape[1]))[0]

def log_probability(parameters,design,observed,uncertainty):
    p=np.atleast_2d(parameters);res=(observed[None]-p@design.T)/uncertainty
    value=-.5*np.sum(res**2+np.log(2*np.pi*uncertainty**2),axis=1)
    value[(p[:,0]<0)|(p[:,1]<0)]=-np.inf
    return value

def stretch_proposal(active,complement,random,scale=2.):
    n,ndim=active.shape;z=((scale-1)*random.rand(n)+1)**2/scale;partners=complement[random.randint(len(complement),size=n)]
    return partners-(partners-active)*z[:,None],(ndim-1)*np.log(z)

def ensemble_step(positions,logp,design,observed,uncertainty,random):
    positions=positions.copy();logp=logp.copy();walkers=len(positions);accepted=np.zeros(walkers,dtype=bool);labels=np.arange(walkers)%2;random.shuffle(labels)
    for split in [0,1]:
        mask=labels==split;ids=np.flatnonzero(mask);proposal,factors=stretch_proposal(positions[mask],positions[~mask],random);new=log_probability(proposal,design,observed,uncertainty);accept=factors+new-logp[mask]>np.log(random.rand(len(ids)));chosen=ids[accept];positions[chosen]=proposal[accept];logp[chosen]=new[accept];accepted[chosen]=True
    return positions,logp,accepted

def sample_posterior(initial,design,observed,uncertainty,seed=20261025,steps=4000,walkers=32):
    ndim=len(initial);global_rng=np.random.RandomState(seed);random=np.random.RandomState(seed);positions=initial+1e-5*global_rng.randn(walkers,ndim);logp=log_probability(positions,design,observed,uncertainty);chain=np.empty((steps,walkers,ndim));accepted=np.zeros(walkers,dtype=int)
    for step in range(steps):
        random.choice(1,p=[1.]) # original move-selection RNG consumption
        positions,logp,acc=ensemble_step(positions,logp,design,observed,uncertainty,random);accepted+=acc
        chain[step]=positions
    return chain,accepted/steps

def autocorrelation_time(chain):
    """Ensemble-mean normalized ACF, Sokal self-consistent window c=5."""
    n=len(chain);x=chain-chain.mean(axis=0,keepdims=True);size=1<<(2*n-1).bit_length();f=np.fft.fft(x,n=size,axis=0);acf=np.fft.ifft(f*np.conjugate(f),axis=0).real[:n];acf/=acf[0:1];acf=acf.mean(axis=1);taus=2*np.cumsum(acf,axis=0)-1;out=[]
    for j in range(chain.shape[-1]):
        candidates=np.flatnonzero(np.arange(n)>=5*taus[:,j]);k=int(candidates[0]) if len(candidates) else n-1;out.append(taus[k,j])
    return np.array(out)

def reconstruct(data,seed=20261025,steps=4000,burn=400,components=3):
    lightcurves,areas=paired_lightcurves(data['time']);coeff,scores,latent=eigencurves(lightcurves);design=design_matrix(lightcurves,scores,components)
    initial=initialize_parameters(design,data['measurements'])
    chain,acceptance=sample_posterior(initial,design,data['measurements'],data['uncertainty'],seed,steps);retained=chain[burn:];mean=retained.mean(axis=(0,1));sh=spherical_coefficients(mean,coeff);image=harmonics(data['latitude'],data['longitude'])@sh;prediction=design@mean;tau=autocorrelation_time(retained)
    return image,{'prediction':prediction,'mean_parameters':mean,'spherical_coefficients':sh,'chain':retained,'autocorrelation':tau,'effective_sample_size':retained.shape[0]*retained.shape[1]/tau,'acceptance':acceptance,'initial':initial,'design':design,'pca_coefficients':coeff,'scores':scores,'eigenvalues':latent,'lightcurves':lightcurves,'areas':areas}
