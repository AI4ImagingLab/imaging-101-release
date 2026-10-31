"""Local Gaussian fluorescence HMM and scaled sequence-boundary operators."""
import numpy as np
from scipy.special import digamma
TINY=1e-300
def gaussian_probability(observations,means,variance):
    x=np.asarray(observations)
    return np.exp(-.5*np.log(2*np.pi)-.5*np.log(variance)-.5*(x[...,None]-means)**2/variance)+TINY
def variational_emissions(observations,a,b,beta,means):
    elog_precision=digamma(a)-np.log(b)
    distance=1/beta+a/b*(observations[...,None]-means)**2
    return np.maximum((2*np.pi)**(-.5)*np.exp(-.5*(distance-elog_precision)),TINY)
def forward_backward_batch(emission,transition,initial,lengths):
    n,t,k=emission.shape;valid=np.arange(t)[None,:]<lengths[:,None]
    alpha=np.zeros_like(emission);back=np.ones_like(emission);scales=np.ones((n,t))
    value=initial*emission[:,0];scales[:,0]=value.sum(axis=1);alpha[:,0]=value/(scales[:,0,None]+TINY)
    for j in range(1,t):
        value=(alpha[:,j-1]@transition)*emission[:,j];total=value.sum(axis=1)
        alpha[:,j]=np.where(valid[:,j,None],value/(total[:,None]+TINY),alpha[:,j-1]);scales[:,j]=np.where(valid[:,j],total,1.)
    for j in range(t-2,-1,-1):
        value=(emission[:,j+1]*back[:,j+1])@transition.T/(scales[:,j+1,None]+TINY)
        back[:,j]=np.where(valid[:,j+1,None],value,1.)
    posterior=alpha*back*valid[:,:,None];xi_sum=np.zeros((k,k))
    for j in range(t-1):
        value=alpha[:,j,:,None]*transition[None]*emission[:,j+1,None,:]*back[:,j+1,None,:]/(scales[:,j+1,None,None]+TINY)
        xi_sum+=(value*valid[:,j+1,None,None]).sum(axis=0)
    return posterior,xi_sum,float(np.sum(np.log(scales+TINY)))
def variational_step(observations,lengths,a,b,beta,means,pi,transition_posterior):
    elog_transition=digamma(transition_posterior)-digamma(transition_posterior.sum(axis=1))[:,None]
    elog_initial=digamma(pi)-digamma(pi.sum())
    prob=variational_emissions(observations,a,b,beta,means)
    return forward_backward_batch(prob,np.exp(elog_transition),np.exp(elog_initial),lengths)
def viterbi_sequence(observations,means,variance,transition,initial):
    logp=np.log(gaussian_probability(np.asarray(observations),means,variance));A=np.log(transition/transition.sum(axis=1,keepdims=True));start=np.log(initial)
    scores=np.empty_like(logp);pointers=np.zeros_like(logp,dtype=np.int64);scores[0]=start+logp[0]
    for i in range(1,len(observations)):
        pair=scores[i-1,:,None]+A;pointers[i]=np.argmax(pair,axis=0);scores[i]=logp[i]+np.max(pair,axis=0)
    path=np.empty(len(observations),dtype=np.int64);path[-1]=int(np.argmax(scores[-1]))
    for i in range(len(path)-2,-1,-1):path[i]=pointers[i+1,path[i+1]]
    return path
def model_log_likelihood(sequences,means,variance,transition,initial):
    total=0.
    for x in sequences:
        prob=gaussian_probability(x,means,variance);forward=initial*prob[0];scale=forward.sum();forward/=scale+TINY;total+=np.log(scale+TINY)
        for row in prob[1:]:
            forward=(forward@transition)*row;scale=forward.sum();forward/=scale+TINY;total+=np.log(scale+TINY)
    return float(total)
