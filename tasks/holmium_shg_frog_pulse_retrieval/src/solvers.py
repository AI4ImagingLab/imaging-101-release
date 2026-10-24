"""Local COPRA/PIE/power-PCGPA updates, MIT-derived from pypret d47deb6.

Copyright (c) 2019 Nils C. Geib.
General numerical primitives only; no upstream imports.
"""
import time
import numpy as np
from .physics_model import FourierGrid,SHGFrog,abs2,norm2,scaled_residual

class Retriever:
    def __init__(self,model,measured,initial,method='copra',alpha=.25,seed=1234):
        self.model=model;self.measured=measured;self.method=method;self.alpha=alpha
        self.rng=np.random.RandomState(seed);self.spectrum=initial.copy()
        signal,a,e,h=model.signal(initial);trace=abs2(model.grid.forward(signal))
        self.best_error,self.mu,_=scaled_residual(measured,trace)
        self.best=initial.copy();self.best_approx=False;self.approx=False
        self.mode='local';self.stale=0;self.current_max=0.
        if method=='copra':
            target=model.project(measured/self.mu,signal)
            grad=model.gradient(target,signal,a,e,h)
            self.current_max=float(np.max(np.sum(abs2(grad),axis=1)))
        if method=='pcgpa' and not np.array_equal(model.delays,model.grid.t):
            raise ValueError('PCGPA comparison requires exactly the Fourier delay grid')
    def step(self,mode=None,capture=False):
        m=self.model;g=m.grid;x=self.spectrum.copy();d=self.measured;out={}
        self.previous_max=self.current_max;self.current_max=0.
        if self.stale==5:self.mode='global'
        if mode is not None:self.mode=mode
        if self.method in ['copra','pie'] and (self.method=='pie' or self.mode=='local'):
            T=np.zeros(d.shape)
            if self.method=='pie':beta=self.rng.uniform(.1,.5);out['beta']=beta
            order=self.rng.permutation(np.arange(d.shape[0]));out['order']=order
            for i in order:
                signal,a,e,h=m.signal(x,int(i));T[i]=abs2(g.forward(signal))
                target=m.project(d[i]/self.mu,signal)
                if self.method=='copra':
                    grad=m.gradient(target,signal,a,e,h);z=norm2(target-signal)
                    self.current_max=max(norm2(grad),self.current_max)
                    gamma=z/max(self.current_max,self.previous_max);x-=gamma*grad
                    if capture:out.update(Smk2=target,nablaZnm=grad,Zm=z,gamma=gamma)
                else:
                    e+=beta*a.conj()*(target-signal)/abs2(e).max();x[:]=g.forward(e)
                    if capture:out['Smk2']=target
            self.approx=True;R,self.mu,r=scaled_residual(d,T)
        elif self.method=='copra':
            signal,a,e,h=m.signal(x);T=abs2(g.forward(signal));R,self.mu,r=scaled_residual(d,T)
            self.approx=False
            grad_r=-4*g.dt/(g.dw*2*np.pi)*g.backward(self.mu*g.forward(signal)*(d-self.mu*T))
            eta_r=self.alpha*r/norm2(grad_r);target=signal-eta_r*grad_r
            grad=m.gradient(target,signal,a,e,h).sum(axis=0);z=norm2(target-signal)
            eta_z=self.alpha*z/norm2(grad);x-=eta_z*grad
            if capture:out.update(r=r,gradrmk=grad_r,etar=eta_r,Smk2=target,nablaZn=grad,Z=z,etaz=eta_z)
        elif self.method=='pcgpa':
            signal,a,e,h=m.signal(x);T=abs2(g.forward(signal));R,self.mu,r=scaled_residual(d,T)
            target=m.project(d/self.mu,signal)
            for j in range(g.n):target[:,j]=np.roll(target[::-1,j],j)
            e=target.conj()@e;e=e.conj()/np.sqrt(norm2(e));x[:]=g.forward(e);self.approx=False
        else:raise ValueError('Unknown method')
        # Deliberately preserve upstream OLD-input best-save convention.
        if R<self.best_error:
            self.best_error=R;self.best=self.spectrum.copy();self.best_approx=self.approx;self.stale=0
        else:self.stale+=1
        self.spectrum=x
        if capture:out.update(Tmn=T,spectrum=x.copy(),reported_error=R,mu=self.mu,approximate_error=self.approx,current_max_gradient=self.current_max,previous_max_gradient=self.previous_max,signal_before=signal)
        return R,out
    def run(self,iterations=1000):
        start=time.perf_counter();errors=[];modes=[];R=self.best_error
        for i in range(iterations):
            # Preserve source logging: replace a local mixed-row statistic
            # with the complete residual of the current field;
            # global-mode log is the preceding input's complete trace error.
            if self.approx:R=scaled_residual(self.measured,self.model.trace(self.spectrum))[0]
            errors.append(R)
            R,_=self.step();modes.append(0 if self.mode=='local' else 1)
        actual=scaled_residual(self.measured,self.model.trace(self.best))[0]
        return dict(spectrum=self.best.copy(),trace_error=np.array(errors),mode=np.array(modes),actual_error=actual,runtime_seconds=time.perf_counter()-start)
