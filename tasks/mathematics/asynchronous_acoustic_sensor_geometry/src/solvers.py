"""Explicit source Gauss-Newton updates with reproducible uniform restarts."""
import numpy as np
from .physics_model import pack,unpack,residual_jacobian,free_indices,MICROPHONES,EVENTS
VARIANCE=np.r_[np.full(256,1e-8),np.full(39,1e-4)]

def normal_step(jacobian,residual):
    r"""Preserve source left-to-right normal-equation arithmetic.

    Original -(J'/W*J)\J'/W*r solves against all J' columns before W/r.
    W is diagonal; elementwise right division evaluates that exact operator.
    """
    normal=(jacobian.T/VARIANCE)@jacobian
    return (np.linalg.solve(-normal,jacobian.T)/VARIANCE)@residual

def uniform_starts(seed=20261029,count=30):
    """Portable RNG realization of source real-experiment uniform spatial prior.

    Independent of observations and targets. Replaces only MATLAB RNG stream;
    exact same arrays were supplied to the independent author GN oracle.
    """
    rng=np.random.default_rng(seed)
    for _ in range(count):
        state=np.zeros((MICROPHONES+EVENTS,5))
        state[:,:3]=rng.uniform([-.8,0.,-1.],[.8,2.,0.],size=(24,3))
        state[10,:3]=0;state[11,1:3]=0;state[12,2]=0
        yield state

def gauss_newton(initial,data,max_iterations=100,fit_clocks=True):
    """Original full step/stop/divergence rules, plus a portable iteration cap.

    Native author wall-clock cap is hardware dependent. Every accepted oracle
    candidate terminates by the same update/cost criteria before that cap.
    """
    state=initial.copy();r,j=residual_jacobian(state,data);first_cost=.5*np.dot(r/VARIANCE,r)
    history=[];converged=False
    for iteration in range(max_iterations):
        r,j=residual_jacobian(state,data)
        try:
            if fit_clocks:step=normal_step(j,r)
            else:
                geometry=free_indices()%5<3;step=np.zeros(j.shape[1]);step[geometry]=normal_step(j[:,geometry],r)
        except np.linalg.LinAlgError:break
        cost=.5*np.dot(r/VARIANCE,r);size=np.linalg.norm(step)
        state=unpack(pack(state)+step)
        history.append((iteration,float(cost),float(size)))
        if not np.isfinite(state).all():break
        if size<.01 or cost<30:
            converged=True;break
        if not np.isfinite(cost) or cost/max(first_cost,np.finfo(float).tiny)>1e5:break
    r,j=residual_jacobian(state,data)
    return state,{'converged':converged,'cost':float(.5*np.dot(r/VARIANCE,r)),'history':history,'iterations':len(history)}

def reconstruct(data,seed=20261029,starts=30,fit_clocks=True):
    """Complete bound ten-sensor/fourteen-event joint geometry and clock fit."""
    best=None;table=[];successful=0
    for index,state in enumerate(uniform_starts(seed,starts)):
        try:estimate,info=gauss_newton(state,data,fit_clocks=fit_clocks)
        except (FloatingPointError,ValueError):
            table.append({'start':index+1,'converged':False,'cost':None});continue
        usable=info['converged'] and np.isfinite(info['cost'])
        table.append({'start':index+1,'converged':bool(usable),'cost':info['cost'] if np.isfinite(info['cost']) else None,'iterations':info['iterations']})
        if usable:
            successful+=1
            if best is None or info['cost']<best[1]['cost']:best=(estimate,info,index+1)
    if best is None:raise RuntimeError('No original convergence criterion reached in any independent start')
    estimate,info,selected=best
    return {'reconstruction':estimate[:,:3][None]},dict(info,state=estimate,selected_start=selected,successful_starts=successful,starts=table)
