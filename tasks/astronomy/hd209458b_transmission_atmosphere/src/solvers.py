"""Explicit Gaussian atmospheric likelihood and affine-invariant stretch updates.

Published PLATON v4 prior/initialization conventions (GPL-3.0), with the
Goodman-Weare stretch / red-blue transition as implemented by emcee3.1.6 (MIT).
No domain reconstruction or external MCMC sampler is imported.
"""
import numpy as np
from scipy.stats import norm
from .physics_model import R_SUN,R_JUP,M_JUP,AtmosphereError

GUESS=np.array([1.19*R_SUN,.73*M_JUP,1.4*R_JUP,1200.,0.,0.,4.,1.])
GAUSSIAN_STD=np.array([.02*R_SUN,.04*M_JUP])
LOW=np.array([-np.inf,-np.inf,0.,300.,0.,-1.,-.99,0.])
HIGH=np.array([np.inf,np.inf,np.inf,3000.,5.,3.,5.,np.inf])
INITIAL_LOW=np.array([GUESS[0]-2*GAUSSIAN_STD[0],GUESS[1]-2*GAUSSIAN_STD[1],.9*GUESS[2],600.,0.,-1.,-.99,.5])
INITIAL_HIGH=np.array([GUESS[0]+2*GAUSSIAN_STD[0],GUESS[1]+2*GAUSSIAN_STD[1],1.1*GUESS[2],1800.,1.,3.,5.,5.])

def log_prior(parameters):
    """Original two Gaussian ancillary constraints and six strict uniform bounds."""
    p=np.asarray(parameters)
    result=0.
    for j in range(8):
        if j<2:result+=np.log(norm.pdf(p[j],GUESS[j],GAUSSIAN_STD[j]))
        elif not (p[j]>LOW[j] and p[j]<HIGH[j]):return -np.inf
    return float(result)

def log_probability(parameters,model):
    """Observed-depth Gaussian likelihood with inferred multiplicative error scale."""
    prior=log_prior(parameters)
    if not np.isfinite(prior):return -np.inf
    if parameters[0]<=0 or parameters[1]<=0:return -np.inf
    try:prediction=model.forward(parameters)
    except AtmosphereError:return -np.inf
    residual=prediction-model.raw['transit_depth'];errors=parameters[7]*model.raw['uncertainty']
    likelihood=-.5*np.sum(residual**2/errors**2+np.log(2*np.pi*errors**2))
    return float(prior+likelihood)

def initialize_walkers(seed=20261028,walkers=50):
    """Original walker-major draw order; first walker is the fiducial guess."""
    random=np.random.RandomState(seed);positions=[]
    for i in range(walkers):
        row=[]
        for j in range(8):row.append(GUESS[j] if i==0 else random.uniform(INITIAL_LOW[j],INITIAL_HIGH[j]))
        positions.append(row)
    return np.asarray(positions),random

def stretch_chain(model,steps=1000,walkers=50,seed=20261028,progress=None):
    """Complete selected sampler: two randomized red/blue groups and a=2 stretch.

    Returns chain(walkers,steps,8), logp(walkers,steps), acceptance(walkers).
    The standard protocol keeps all1000 samples per walker, with no burn discard.
    Smaller steps are only used by independently frozen transition unit tests.
    """
    if walkers<16:raise ValueError('Require at least twice as many walkers as dimensions')
    positions,random=initialize_walkers(seed,walkers);logp=np.array([log_probability(p,model) for p in positions]);chain=np.empty((walkers,steps,8));probability=np.empty((walkers,steps));acceptances=np.zeros(walkers,dtype=np.int64);indices=np.arange(walkers)
    for step in range(steps):
        # The original one-element move selection still consumes one RNG draw.
        random.choice(1,p=np.array([1.]))
        groups=indices%2;random.shuffle(groups);accepted=np.zeros(walkers,dtype=bool)
        for split in range(2):
            subset=groups==split;sets=[positions[groups==j] for j in range(2)];current=sets[split];complement=np.concatenate(sets[:split]+sets[split+1:],axis=0)
            stretch=((2.-1.)*random.rand(len(current))+1)**2./2.
            factors=(8.-1.)*np.log(stretch);partners=random.randint(len(complement),size=(len(current),))
            proposed=complement[partners]-(complement[partners]-current)*stretch[:,None]
            proposed_logp=np.array([log_probability(p,model) for p in proposed])
            for j,f,new_logp in zip(indices[subset],factors,proposed_logp):
                with np.errstate(invalid='ignore'):difference=f+new_logp-logp[j]
                if difference>np.log(random.rand()):accepted[j]=True
            update=subset&accepted;proposed_update=accepted[subset]
            positions[update]=proposed[proposed_update];logp[update]=proposed_logp[proposed_update]
        acceptances+=accepted;chain[:,step]=positions;probability[:,step]=logp
        if progress is not None and (step+1)%10==0:progress(step+1,float(np.max(logp)))
    return chain,probability,acceptances/float(steps)

def reconstruct(model,seed=20261028,steps=1000,progress=None):
    """Original complete default50x1000 sampler and original median/best summaries."""
    chain,logp,acceptance=stretch_chain(model,steps=steps,seed=seed,progress=progress)
    samples=chain.reshape(-1,8);flat_logp=logp.reshape(-1);best=samples[np.argmax(flat_logp)]
    median=np.median(samples,axis=0);depth=model.forward(best)
    return dict(reconstruction=median[None],best_fit=best[None],transit_depth=depth[None]),dict(chain=chain,log_probability=logp,acceptance_fraction=acceptance)
