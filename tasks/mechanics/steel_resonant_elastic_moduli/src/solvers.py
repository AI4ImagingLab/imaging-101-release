"""Original RUScal LM updates plus explicit manual-correspondence adaptation.

The author implementation accepts a manually paired list. Here monotone dynamic
programming performs that matching using measured frequencies and calculated
spectra only. Original manual assignments are confined to independent fixtures.
"""
import numpy as np
from .physics_model import resonances

def match_modes(measured,calculated):
    """Match every observed frequency in order, allowing missing model modes."""
    m,n=len(measured),len(calculated)
    if m>n:raise ValueError('Fewer theoretical than observed modes')
    cost=(np.asarray(measured)[:,None]/np.asarray(calculated)[None,:]-1)**2
    dp=np.full((m+1,n+1),np.inf);dp[0,:]=0
    take=np.zeros_like(dp,dtype=bool)
    for i in range(1,m+1):
        for j in range(i,n+1):
            assigned=dp[i-1,j-1]+cost[i-1,j-1]
            skipped=dp[i,j-1]
            if assigned<=skipped:dp[i,j]=assigned;take[i,j]=True
            else:dp[i,j]=skipped
    indices=np.empty(m,dtype=np.int64);i,j=m,n
    while i:
        if take[i,j]:indices[i-1]=j-1;i-=1
        j-=1
    return indices

def objective(observed,calculated,weights):
    """RUScal percent-squared mean; count nonzero weights, not their sum."""
    return float(np.sum(weights*(observed/calculated-1)**2)/(.0001*np.count_nonzero(weights)))

def paired_lm(observed,weights,indices,initial):
    """Preserve source frequency normal equations and relative acceptance."""
    parameters=np.array(initial,dtype=np.float64)
    delta=parameters*.001  # fixed at initial, as in original main()
    damping=.0001;history=[]
    for iteration in range(20):
        predicted=resonances(parameters)[indices]
        before=objective(observed,predicted,weights)
        derivative=[]
        for j in range(2):
            perturbed=parameters.copy();perturbed[j]+=delta[j]
            derivative.append((resonances(perturbed)[indices]-predicted)/delta[j])
        jac=np.asarray(derivative).T
        alpha=jac.T@(weights[:,None]*jac)
        beta=jac.T@(weights*(observed-predicted))
        diagonal=np.sqrt(np.diag(alpha))
        correlation=alpha/(diagonal[:,None]*diagonal[None,:])
        for attempt in range(30):
            modified=correlation.copy();modified.flat[::3]*=1+damping
            inverse=np.linalg.inv(modified)
            update=np.sum(beta[None,:]*inverse/(diagonal[:,None]*diagonal[None,:]),axis=1)
            trial=parameters+update
            trial_frequency=resonances(trial)[indices]
            after=objective(observed,trial_frequency,weights)
            if before-after<=-1e-5:damping*=10
            else:break
        else:raise RuntimeError('Source damping search did not accept a step')
        parameters=trial
        history.append({'iteration':iteration,'moduli_mbar':trial.tolist(),'before':before,'after':after,'damping':damping})
        damping/=10
        if 1-after/before<1e-5:break
    return parameters,history

def reconstruct(observed,weights,initial=(2.7,.8)):
    """Initialize on twelve low modes, then infer and fit all correspondences.

    The low-mode initialization implements the paper's recommendation to fit a
    few low-frequency resonances first. It is a universal initialization rule,
    not a stored per-observation mode map. Full inversion retains every supplied
    measured positive frequency, including duplicates and author zero weights.
    """
    observed=np.asarray(observed,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    parameters,initial_history=paired_lm(observed[:12],weights[:12],np.arange(12),initial)
    histories=[];previous=None
    for outer in range(10):
        calculated=resonances(parameters)
        indices=match_modes(observed,calculated)
        parameters,history=paired_lm(observed,weights,indices,parameters)
        histories.append({'matching_pass':outer,'indices':indices.tolist(),'fit':history})
        if previous is not None and np.array_equal(indices,previous):break
        previous=indices.copy()
    else:raise RuntimeError('Mode correspondences failed to stabilize')
    frequencies=resonances(parameters)
    return {'reconstruction':(parameters*100)[None],'resonances':frequencies[None]}, {'initial_history':initial_history,'history':histories,'mode_indices':indices.tolist(),'weighted_rms_percent':np.sqrt(objective(observed,frequencies[indices],weights))}
