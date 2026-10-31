"""CC BY 4.0 adaptation of XPSinv cp_quad.jl by Matthew Ozon,
Konstantin Tumashevich and Nønne L. Prisle (2022).
Explicit accelerated Chambolle–Pock updates; diagonal covariance specialization.
"""
import numpy as np
from .physics_model import second_difference
def dual_prox(values,target,sigma,covariance_diagonal):
    return (values-sigma*target)/(1+.5*sigma*covariance_diagonal)
def reconstruct(system_matrix,observations,iterations=100000,prior_factor=1.,save_every=1000):
    """Return profile and numeric diagnostics, estimating every one of 201 samples.

    The variance of each original acquisition is .01² µm⁶. All repeats
    enter their mean. Prior covariance is (N/M)*.01² times prior_factor.
    No endpoint or concentration value is supplied to the algorithm.
    """
    h=np.asarray(system_matrix);y=np.mean(observations,axis=0);n=h.shape[1];m=h.shape[0]
    d=second_difference(n);a=np.vstack((h,d));target=np.r_[y,np.zeros(n-2)]
    covariance=np.r_[np.full(m,.01**2/len(observations)),np.full(n-2,n/m*.01**2*prior_factor)]
    tau=10.;sigma=1/(tau*np.linalg.norm(a,2)**2);gamma=2*np.sqrt(n)/tau
    x=np.full(n,.5);bar=x.copy();s=np.zeros(len(target));xs=[x.copy()];ss=[s.copy()];ts=[tau];history=[]
    for step in range(1,iterations+1):
        s=dual_prox(s+sigma*(a@bar),target,sigma,covariance)
        old=x;x=np.maximum(x-tau*(a.T@s),0)
        theta=1/np.sqrt(1+2*gamma*tau);tau*=theta;sigma/=theta;bar=x+theta*(x-old)
        rel=np.median(np.abs(y-h@x)/y);change=np.linalg.norm(x-old)/np.linalg.norm(x)
        if step<=50:xs.append(x.copy());ss.append(s.copy());ts.append(tau)
        if step%save_every==0 or step==1:
            residual=a@x-target;history.append((step,float(np.sum(residual**2/covariance)),rel,change))
        if rel<=1e-4 and change<=1e-9:break
    return x,{'iterations':step,'tau':tau,'primal_path':np.array(xs).T,'dual_path':np.array(ss).T,'tau_path':np.array(ts),'history':np.array(history),'dual':s}
