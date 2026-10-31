"""All600-frame assembly and explicit positivity-constrained IRLS."""
import time
import numpy as np
from .preprocessing import element_deformation
from .physics_model import features,weak_operator,frame_system
PENALTY=10000.
def assemble(data,progress=None):
    H=np.zeros((171,171)); b=np.zeros(171); C=0.
    for i,u in enumerate(data['displacements']):
        F=element_deformation(data['triangles'],data['shape_gradients'],u)
        _,G=features(F)
        A=weak_operator(data['triangles'],data['shape_gradients'],data['element_areas'],G,len(data['coordinates']))
        h,r,c=frame_system(A,data['free_dofs'],data['boundary_masks'],data['reactions'][i])
        H+=h; b+=r; C+=c
        if progress is not None: progress(i+1)
    return H,b,C
def positive_quadratic(M,v,tol=1e-12):
    n=len(v); x=np.zeros(n); passive=np.zeros(n,dtype=bool)
    for iteration in range(5000):
        g=M@x-v
        if np.min(g[~passive],initial=0)>=-tol:return x,iteration
        passive[np.argmin(np.where(~passive,g,np.inf))]=True
        for inner in range(n+1):
            ids=np.flatnonzero(passive); trial=np.zeros(n)
            trial[ids]=np.linalg.solve(M[np.ix_(ids,ids)],v[ids])
            bad=passive&(trial<=0)
            if not np.any(bad):x=trial;break
            alpha=np.min(x[bad]/(x[bad]-trial[bad])); x+=alpha*(trial-x)
            remove=passive&(x<=1e-15); x[remove]=0;passive[remove]=False
        else:raise RuntimeError('Active-set inner failure')
    raise RuntimeError('Active-set convergence failure')
def solve(H,b,C,seed=0,penalty=PENALTY):
    s=np.sqrt(np.diag(H)); T=np.sqrt(C); Q=H/s[:,None]/s[None,:]
    r=penalty/s/T; v=b/s/T
    u=np.random.RandomState(seed).uniform(.01,.1,len(b)); history=[]
    for floor in [1e-3,1e-5,1e-7,1e-9,1e-11,1e-13]:
        for iteration in range(3000):
            new,_=positive_quadratic(Q+np.diag(r/np.maximum(u,floor)),v)
            change=np.max(abs(new-u));u=new
            if change<1e-12:break
        else:raise RuntimeError('IRLS did not converge')
        history.append((floor,iteration+1,float(change)))
    g=Q@u+r-v; kkt=float(np.max(abs(np.minimum(u,g))))
    if kkt>1e-8:raise RuntimeError('Full projected KKT failed')
    return u*T/s,dict(history=history,projected_kkt=kkt,seed=seed)
def objective(theta,H,b,C,penalty=PENALTY):
    weak=float(.5*theta@H@theta-b@theta+C)
    return dict(weak_sse=weak,penalty=float(penalty*np.sum(abs(theta))),total=float(weak+penalty*np.sum(abs(theta))),weak_residual_relative_l2=float(np.sqrt(max(0,weak)/C)))
def reconstruct(data,seed=0):
    start=time.perf_counter(); H,b,C=assemble(data);theta,info=solve(H,b,C,seed)
    info.update(objective(theta,H,b,C)); info['seconds']=time.perf_counter()-start
    return theta,info,(H,b,C)
