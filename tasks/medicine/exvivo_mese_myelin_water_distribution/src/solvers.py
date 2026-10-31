"""Explicit nonnegative active-set least squares and objective certificates."""
import numpy as np

def active_set_nnls(A,y,ridge=0.,tolerance=1e-11):
    """Lawson-Hanson feasible active-set updates; general dense solves only."""
    n=A.shape[1];G=A.T@A+ridge*np.eye(n);b=A.T@y;x=np.zeros(n);P=np.zeros(n,bool)
    threshold=tolerance*max(1.,float(np.max(abs(b))));steps=0
    while True:
        w=b-G@x;candidate=np.where(~P,w,-np.inf)
        if np.max(candidate)<=threshold:return x
        P[np.argmax(candidate)]=True
        while True:
            z=np.zeros(n)
            z[P]=np.linalg.solve(G[np.ix_(P,P)],b[P]) if ridge else np.linalg.lstsq(A[:,P],y,rcond=None)[0]
            if np.all(z[P]>0):x=z;break
            bad=P&(z<=0);ratios=x[bad]/(x[bad]-z[bad]);alpha=float(np.min(ratios))
            x+=alpha*(z-x);drop=P&(x<=1e-13);x[drop]=0.;P[drop]=False
            steps+=1
            if steps>1000:raise RuntimeError('Nonnegative active-set inner loop failed to converge')
        steps+=1
        if steps>1000:raise RuntimeError('Nonnegative active-set failed to converge')

def solve_distribution(dictionary,y,labels,ridge):
    result=np.empty((len(y),dictionary.shape[-1]))
    for c,A in enumerate(dictionary):
        for i in np.flatnonzero(labels==c):result[i]=active_set_nnls(A,y[i],ridge)
    return result

def objective_and_gap(dictionary,x,y,labels,ridge):
    """Primal SSE+ridge, stable nonnegative Fenchel primal-dual gap."""
    objective=np.empty(len(y));gap=np.empty(len(y))
    for c,A in enumerate(dictionary):
        m=labels==c;u=x[m]@A.T-y[m];g=u@A
        objective[m]=np.sum(u*u,axis=1)+ridge*np.sum(x[m]*x[m],axis=1)
        gap[m]=np.sum((np.sqrt(ridge)*x[m]+np.minimum(g,0)/np.sqrt(ridge))**2,axis=1)+2*np.sum(x[m]*np.maximum(g,0),axis=1)
    return objective,gap
