"""Source bisection initialization and relative-coordinate inexact Gauss-Newton."""
import numpy as np
from scipy.sparse.linalg import splu
from .preprocessing import geometry_operators
from .physics_model import (assemble_fem,continuum_prejacobian,forward_fluorescence,MUAX,MUAM,KAPPAX,KAPPAM,ETA)

def initialize_uniform(observed,phix,mass,emission_lu,detector,links,max_iterations=40,tolerance=1e-4):
    """Original finite-difference derivative bisection; returns muaf [mm^-1]."""
    n=len(phix);log_observed=np.log(observed);history=[]
    def objective(muaf):
        predicted,_=forward_fluorescence(np.full(n,ETA*muaf),phix,mass,emission_lu,detector,links)
        return np.sum((np.log(predicted)-log_observed)**2)
    a=1e-10;b=.1;delta=1e-10
    ea=objective(a);da=(objective(a+delta)-ea)/delta
    eb=objective(b);db=(objective(b+delta)-eb)/delta
    for iteration in range(max_iterations):
        c=a+(b-a)/2;ec=objective(c);dc=(objective(c+delta)-ec)/delta
        if dc*da<0 and dc*db>0:b=c;db=dc
        elif dc*db<0 and dc*da>0:a=c;da=dc
        history.append((c,ec,dc))
        if len(history)>1 and abs(history[-1][1]-history[-2][1])<tolerance:break
    return c,np.asarray(history)

def reconstruct(raw,regularization=10.,max_iterations=40,calibrate=True):
    """Recover complete fine-mesh yield with the published unguided pipeline.

    `raw` is the unbatched validated observations/geometry dictionary. Every
    measurement and fine node is retained. Diagnostic settings modify only
    declared regularization or observation calibration for active comparisons.
    """
    if not np.isfinite(regularization) or regularization<=0:raise ValueError('Positive finite regularization required')
    if not isinstance(max_iterations,int) or max_iterations<1:raise ValueError('Positive integer iteration count required')
    source,detector,adjoint,C,B=geometry_operators(raw)
    Ax,mass,vol=assemble_fem(raw['nodes'],raw['tetrahedra'],MUAX,KAPPAX)
    Am,_,_=assemble_fem(raw['nodes'],raw['tetrahedra'],MUAM,KAPPAM)
    phix=splu(Ax).solve(source.toarray()*np.cos(.15));emission_lu=splu(Am)
    aphim=splu(Am.astype(complex)).solve(adjoint.toarray()*np.exp(-1j*.15))
    excitation=(detector@phix)[raw['links'][:,1],raw['links'][:,0]]
    observed=raw['fluorescence']*(excitation/raw['excitation']) if calibrate else raw['fluorescence'].copy()
    initial_muaf,initial_history=initialize_uniform(observed,phix,mass,emission_lu,detector,raw['links'])
    gamma=np.full(len(phix),ETA*initial_muaf);yield_field=gamma.copy()
    pre=continuum_prejacobian(raw['tetrahedra'],vol,phix,aphim,raw['links'])
    log_observed=np.log(observed);objective_history=[];regularization_history=[];fields=[]
    for iteration in range(max_iterations):
        prediction,field=forward_fluorescence(gamma,phix,mass,emission_lu,detector,raw['links'])
        difference=log_observed-np.log(prediction);objective_history.append(np.sum(np.abs(difference**2)))
        if iteration>0 and 100*(objective_history[-2]-objective_history[-1])/objective_history[-2]<=2:break
        jacobian=(pre/prediction[:,None]).real
        coarse_gamma=C@gamma
        normalized=(jacobian@B)*coarse_gamma[None]
        hessian=normalized@normalized.T
        damping=regularization*np.diag(hessian).max();regularization_history.append(damping)
        update=(normalized.T@np.linalg.solve(hessian+np.eye(len(observed))*damping,difference))*coarse_gamma
        coarse_gamma=coarse_gamma+update
        coarse_yield=np.where(coarse_gamma<=0,1e-20,coarse_gamma)
        gamma=B@coarse_gamma;yield_field=B@coarse_yield
        fields.append(yield_field.copy())
    # The source return is the last updated fine etamuaf, before the stopping
    # evaluation. Keep both gamma state and positive source yield convention.
    prediction,field=forward_fluorescence(gamma,phix,mass,emission_lu,detector,raw['links'])
    return {'reconstruction':yield_field,'prediction':prediction,'calibrated_observation':observed,'excitation_prediction':excitation,'initial_muaf':initial_muaf,'initial_history':initial_history,'objective_history':np.asarray(objective_history),'regularization_history':np.asarray(regularization_history),'iterate_fields':np.asarray(fields),'phix':phix,'phim':field,'prejacobian':pre,'gamma':gamma,'forward_relative_L2':float(np.linalg.norm(prediction-observed)/np.linalg.norm(observed))}
