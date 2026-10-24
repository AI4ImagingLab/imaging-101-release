"""Published LM inverse, explicitly extracted from the original algorithm."""
import numpy as np
from scipy.linalg.blas import dsyrk
from .physics_model import conv, julia_square_sum, prescott_transpose_product

def fit(y,t,cp,max_iter=100,trace=None,initial=(0.01,0.01)):
    n=len(y);x=np.tile(np.asarray(initial,dtype=np.float64),(n,1));lam=np.full(n,10.);need=np.ones(n,bool);active=np.ones(n,bool)
    j=np.zeros((n,len(t),2));f=x[:,0,None]*conv(cp,x[:,1],t)-y
    residual=julia_square_sum(f);counts=np.zeros(n,int)
    for iteration in range(max_iter):
        ids=np.flatnonzero(active)
        if not len(ids):break
        fresh=np.flatnonzero(active&need)
        for c in range(2):
            h=6.0554544523933395e-6*np.maximum(1,abs(x[fresh,c]))
            xp=x[fresh].copy();xm=x[fresh].copy();xp[:,c]+=h;xm[:,c]-=h
            fp=xp[:,0,None]*conv(cp,xp[:,1],t)-y[fresh]
            fm=xm[:,0,None]*conv(cp,xm[:,1],t)-y[fresh]
            j[fresh,:,c]=(fp-fm)/(h+h)[:,None]
        need[fresh]=False
        ji=j[ids];fi=f[ids]
        jt=np.ascontiguousarray(ji.transpose(0,2,1))
        dt=np.maximum(julia_square_sum(jt),1e-6)
        jj=np.stack([dsyrk(1.,z.T,trans=1) for z in jt])
        jj[:,1,0]=jj[:,0,1]
        jj[:,0,0]+=lam[ids]*dt[:,0];jj[:,1,1]+=lam[ids]*dt[:,1]
        rhs=-prescott_transpose_product(jt,fi)
        dx=np.linalg.solve(jj,rhs[...,None])[...,0]
        predicted=julia_square_sum(ji[:,:,0]*dx[:,0,None]+ji[:,:,1]*dx[:,1,None]+fi)
        trial=x[ids]+dx
        tf=trial[:,0,None]*conv(cp,trial[:,1],t)-y[ids]
        tr=julia_square_sum(tf)
        with np.errstate(invalid='ignore',divide='ignore'):
            rho=(tr-residual[ids])/(predicted-residual[ids])
        if trace is not None:
            trace.append({'x':x[ids].copy(),'j':ji.copy(),'dtd':dt.copy(),'jj':jj.copy(),'rhs':rhs.copy(),'dx':dx.copy(),'fcur':fi.copy(),'trial_f':tf.copy(),'scalars':np.column_stack([lam[ids],residual[ids],predicted,tr,rho])})
        good=rho>1e-3;acc=ids[good]
        x[acc]=trial[good];f[acc]=tf[good];residual[acc]=tr[good];need[acc]=True
        enlarge=ids[good&(rho>.75)];lam[enlarge]=np.maximum(.1*lam[enlarge],1e-16)
        reject=ids[~good];lam[reject]=np.minimum(10*lam[reject],1e16)
        grad=np.max(abs(prescott_transpose_product(jt,f[ids])),axis=1)
        done=(grad<1e-12)|(np.linalg.norm(dx,axis=1)<1e-8*(1e-8+np.linalg.norm(x[ids],axis=1)))
        counts[ids]+=1;active[ids[done]]=False
    return x,f,counts

def reconstruct(observations,initial=(0.01,0.01),max_iter=100):
    """Reconstruct all3000 pixels; return submission fields and diagnostics.

    Diagnostic rates are the source's unconstrained optimizer coordinates.
    Output maps include its published clipping.  The source's residual>1 branch
    writes zeros but never removes a location from this task's fixed support.
    """
    from .preprocessing import prepare_curves
    r1,ct,t,cp=prepare_curves(observations)
    raw,residual,iterations=fit(ct,t,cp,max_iter=max_iter,initial=initial)
    reduced_sse=julia_square_sum(residual)/(ct.shape[1]-2)
    eps=np.finfo(np.float64).eps
    ktrans=np.clip(raw[:,0],eps,5.0)
    with np.errstate(divide='ignore',invalid='ignore'):
        fraction=np.clip(raw[:,0]/raw[:,1],eps,1.0)
    source_success=reduced_sse<=1.0
    ktrans[~source_success]=0.0;fraction[~source_success]=0.0
    fields={'ktrans':ktrans.reshape((1,60,50),order='F'),
            'extracellular_fraction':fraction.reshape((1,60,50),order='F')}
    if not all(np.isfinite(x).all() for x in fields.values()):
        raise RuntimeError('The published estimator produced nonfinite fields')
    diagnostics={'raw_rates':raw,'residual':residual,'iterations':iterations,
                 'concentration':ct,'effective_relaxation_rate':r1,
                 'reduced_sse':reduced_sse,'source_success':source_success}
    return fields,diagnostics

