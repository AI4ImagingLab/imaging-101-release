"""Explicit TERESA parameter mapping and top-down paired greedy inversion.

Adapted from J. M. Abril-Hernandez, TERESApy (2025), MIT license.
Independent parameter cases are batched; no domain inference library is used.
"""
import numpy as np
from .physics_model import layer_activity
from .preprocessing import canonical_proposals

MEAN_PARAMETERS=np.array([563.,.0385,.19,.21])
FACTORS=np.array([.2,.25,.25,.30])
RESOLUTION=30

def profile_batch(parameters,mass_depth,activity,error,z1,z2,clip_rates=True):
    """Return chi (B,) and profiles (B,5,N): A0, SAR, age, activity, flux.

    The earliest remaining proposal wins exact ties, as in the original strict
    '<' loop. The model's optional rate floor uses global nominal mean SAR.
    """
    p=np.atleast_2d(parameters);batch=len(p);n=len(mass_depth)
    a0=p[:,0,None]*(1+p[:,2,None]*z1)
    rate=p[:,1,None]*(1+p[:,3,None]*z2)
    if clip_rates:rate=np.maximum(rate,.2*MEAN_PARAMETERS[1])
    available=np.ones((batch,n),bool);age=np.zeros(batch)
    result=np.empty((batch,5,n));chi=np.zeros(batch)
    rows=np.arange(batch)
    dm=np.diff(np.r_[0.,mass_depth])
    for k,thickness in enumerate(dm):
        predicted=layer_activity(a0,rate,thickness,age[:,None])
        loss=np.where(available,(predicted-activity[k])**2,np.inf)
        selected=np.argmin(loss,axis=1)
        chosen_a=a0[rows,selected];chosen_w=rate[rows,selected]
        previous=age.copy();age+=thickness/chosen_w
        # Original recomputes duration from accumulated times for its chi.
        chosen_prediction=layer_activity(chosen_a,thickness/(age-previous),thickness,previous)
        chi+=((chosen_prediction-activity[k])/error[k])**2
        result[:,:,k]=np.stack([chosen_a,chosen_w,age,chosen_prediction,chosen_a*chosen_w*10],axis=1)
        available[rows,selected]=False
    return np.sqrt(chi/(n-4)),result

def parameter_grid(resolution=RESOLUTION):
    """Upper endpoints excluded, native nested-loop axis order A,w,sA,sw."""
    axes=[m*(1-f)+np.arange(resolution)*(2*f*m/resolution) for m,f in zip(MEAN_PARAMETERS,FACTORS)]
    return np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,4)

def reconstruct(mass_depth,activity,error,seed=20261029,batch_size=4096):
    """Full native 810000-grid reconstruction and all confidence groups."""
    z1,z2=canonical_proposals(len(mass_depth),seed)
    grid=parameter_grid();scores=np.empty(len(grid))
    for start in range(0,len(grid),batch_size):
        stop=min(start+batch_size,len(grid))
        scores[start:stop]=profile_batch(grid[start:stop],mass_depth,activity,error,z1,z2)[0]
    best_index=int(np.argmin(scores));best=grid[best_index]
    # The published multi-script workflow exchanges formatted decimal text.
    # Preserve that precision before profiling its reported solution/clouds.
    reported=np.array([float(f'{best[0]:.2f}'),*(float(f'{x:.4f}') for x in best[1:])])
    rounded_scores=np.array([float(f'{x:.3f}') for x in scores])
    minimum=float(f'{scores[best_index]:.3f}')
    width=np.sqrt(minimum**2+4.72/(len(mass_depth)-4))-minimum
    boundaries=minimum+np.arange(1,4)*width/3
    groups=np.searchsorted(boundaries,rounded_scores,side='left')
    _,profiles=profile_batch(reported,mass_depth,activity,error,z1,z2,clip_rates=False)
    clouds=[]
    for group in range(3):
        selected=grid[groups==group].copy()
        for column,decimals in enumerate([2,4,4,4]):
            selected[:,column]=np.array([float(f'{x:.{decimals}f}') for x in selected[:,column]])
        clouds.append(selected)
    return profiles[0],{'parameters':reported,'unrounded_parameters':best,'minimum_chi':scores[best_index],'grid':grid,'scores':scores,'cloud_parameters':clouds,'z1':z1,'z2':z2}
