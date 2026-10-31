"""Explicit shared Normal-Gamma/Dirichlet variational HMM updates."""
import numpy as np
from scipy.special import digamma
from .preprocessing import pack_sequences
from .physics_model import gaussian_probability,variational_step,viterbi_sequence
def normal_gamma_update(observations,responsibility,a0,b0,m0,beta0):
    count=responsibility.sum(axis=0)+1e-16
    xbar=(responsibility*observations[:,None]).sum(axis=0)/count
    variance=(responsibility*(observations[:,None]-xbar)**2).sum(axis=0)/count
    beta=beta0+count;means=(beta0*m0+count*xbar)/beta
    a=a0+count/2;b=b0+.5*(count*variance+beta0*count/(beta0+count)*(xbar-m0)**2)
    return a,b,means,beta,count
def initialize(sequences,states=2,seed=20261026):
    x=np.concatenate(sequences);m0=np.percentile(x,np.linspace(0,100,states+2))[1:-1]
    beta0=np.ones(states)*.25;a0=np.ones(states)*2.5;b0=np.ones(states)*.01;pi0=np.ones(states);tm0=np.ones((states,states))
    p=gaussian_probability(x,m0,b0/a0);r=p/(p.sum(axis=1)[:,None]+1e-10)
    a,b,_,beta,_=normal_gamma_update(x,r,a0,b0,m0,beta0)
    # Preserve original heuristic's single N(loc=K,scale=1) draw, not a K-vector.
    means=np.sort(m0+np.random.RandomState(seed).normal(states)/np.sqrt(beta))
    p=gaussian_probability(x,means,b/a);r=p/(p.sum(axis=1)[:,None]+1e-10)
    a,b,_,beta,count=normal_gamma_update(x,r,a,b,means,beta)
    pi=np.sqrt(count+1);tm=tm0.copy();path=viterbi_sequence(x,means,b/a,tm,pi)
    for k in range(states):
        keep=path==k
        if keep.sum()>0:
            pi[k]=float(keep.sum())/len(x)*np.sqrt(float(len(sequences)))
            for j in range(states):
                if k!=j:tm[k,j]=np.sum((path[1:]==j)&(path[:-1]==k))+1
            tm[k]/=np.sum(tm[k])*10.;tm[k,k]=1.-np.sum(tm[k]);tm*=np.sqrt(float(len(sequences)))
    return dict(a=a,b=b,beta=beta,means=means,pi=pi,transition_posterior=tm,priors=(a0,b0,m0,beta0,pi0,tm0))
def reconstruct(sequences,states=2,updates=200,seed=20261026):
    padded,valid,lengths=pack_sequences(sequences);x=np.concatenate(sequences);state=initialize(sequences,states,seed)
    a,b,beta,means,pi,tm=[state[k] for k in ['a','b','beta','means','pi','transition_posterior']];a0,b0,m0,beta0,pi0,tm0=state['priors'];history=[]
    for iteration in range(updates):
        posterior,xi,normalizer=variational_step(padded,lengths,a,b,beta,means,pi,tm)
        r=posterior[valid];history.append(normalizer)
        a,b,means,beta,count=normal_gamma_update(x,r,a0,b0,m0,beta0)
        pi=pi0+posterior[:,0].sum(axis=0);tm=tm0+xi
    order=np.argsort(means);variance=1/np.exp(digamma(a)-np.log(b));fraction=r.mean(axis=0)
    A=tm/tm.sum(axis=1,keepdims=True);start=pi/pi.sum()
    return dict(state_mean=means[order],state_variance=variance[order],transition_probability=A[order][:,order],initial_probability=start[order],population_fraction=fraction[order],posterior=r[:,order],a=a[order],b=b[order],beta=beta[order],transition_posterior=tm[order][:,order],pi=pi[order],updates=updates,normalizer_history=np.asarray(history))
