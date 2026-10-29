"""Selected explicit Rosenbrock23, Hairer initial step and SciML PI controller.

Extracted from OrdinaryDiffEqRosenbrock2.7.5 and OrdinaryDiffEqCore in the
private original environment. Generic ODE mathematics, SciML MIT license.
The original model has three active states and two fixed placeholders; retain
the original five-component error norm and initial-state norm convention.
"""
import numpy as np
from scipy.linalg import lu_factor,lu_solve

def integrate(function,jacobian,time_derivative,initial,final_time,abstol=1e-6,reltol=1e-14):
    y=np.array(initial,dtype=float);time=0.;times=[0.];states=[y.copy()]
    d=1/(2+np.sqrt(2));c32=6+np.sqrt(2);old_error=1e-4
    initial_full=np.r_[y,0.,.1];scale=abstol+abs(initial_full)*reltol
    f0=function(0.,y);full_f0=np.r_[f0,0.,0.]
    d0=np.linalg.norm(initial_full/scale)/np.sqrt(5)
    d1=np.linalg.norm(full_f0/scale)/np.sqrt(5)
    h0=1e-6 if d0<1e-5 or d1<1e-5 else (d0/d1)/100
    h0=min(h0,final_time)
    f1=function(h0,y+h0*f0)
    d2=np.linalg.norm(np.r_[f1-f0,0.,0.]/scale)/np.sqrt(5)/h0
    maximum=max(d1,d2)
    h1=max(1e-6,h0*.001) if maximum<=1e-15 else 10**(-(2+np.log10(maximum))/2)
    h=min(100*h0,h1,final_time);accepted=0;rejected=0
    for iteration in range(100000):
        h=min(h,final_time-time)
        if h<=np.spacing(max(time,1.)):raise RuntimeError('Rosenbrock step underflow')
        proposed,fn,estimate=step(function,jacobian,time_derivative,time,y,h,f0)
        denominator=abstol+np.maximum(abs(y),abs(proposed))*reltol
        error=np.linalg.norm(estimate/denominator)/np.sqrt(5)
        qmax=10000 if accepted==0 else 10
        q11=error**.35 if error else 0.
        q=1/qmax if error==0 else np.clip(q11/(old_error**.2)/.9,1/qmax,5.)
        if np.isfinite(error) and error<=1:
            time+=h;y=proposed;f0=fn;accepted+=1
            times.append(time);states.append(y.copy());old_error=max(error,1e-4)
            if time>=final_time:return np.array(times),np.array(states).T,{'accepted':accepted,'rejected':rejected}
            # RosenbrockAdaptiveAlgorithm inherits adaptive-implicit deadband.
            if 1<=q<=1.2:q=1.
            h/=q
        else:
            rejected+=1;h/=min(5.,q11/.9) if np.isfinite(q11) else 5.
    raise RuntimeError('Rosenbrock maximum iteration count exceeded')

def step(function,jacobian,time_derivative,time,y,h,f0=None):
    """One original Rosenbrock23 step and embedded error estimate."""
    if f0 is None:f0=function(time,y)
    d=1/(2+np.sqrt(2));c32=6+np.sqrt(2);gamma_h=h*d;J=jacobian(time,y);ft=time_derivative(time,y)
    factor=lu_factor(np.eye(len(y))-gamma_h*J,check_finite=False)
    k1=lu_solve(factor,f0+gamma_h*ft,check_finite=False)
    midpoint=y+(h/2)*k1;fm=function(time+h/2,midpoint)
    k2=lu_solve(factor,fm-k1,check_finite=False)+k1
    proposed=y+h*k2;fn=function(time+h,proposed)
    k3=lu_solve(factor,fn-c32*(k2-fm)-2*(k1-f0)+h*ft,check_finite=False)
    estimate=h/6*(k1-2*k2+k3)
    return proposed,fn,estimate
