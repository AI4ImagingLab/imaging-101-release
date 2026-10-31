"""Onion-peeling, edge correction and original uncertainty propagation; GPLv3."""
import numpy as np
from .physics_model import projection_volumes,edge_correction,count_to_hydrogen_emission_measure,ARCMIN_KPC,KPC_CM
from .preprocessing import background_subtraction

def median_smooth(profile):
    """Original five-neighbor median and nonperiodic endpoint extrapolation."""
    x=np.column_stack([np.roll(profile,k) for k in [2,1,0,-1,-2]])
    smooth=np.median(x,axis=1)
    smooth[1]=np.median(x[1,1:]);smooth[-2]=np.median(x[-2,:-1])
    smooth[0]=np.median([3*profile[0]-2*profile[1],profile[0],profile[1]])
    smooth[-1]=np.median([3*profile[-1]-2*profile[-2],profile[-2],profile[-1]])
    return smooth

def onion_peeling(profile,error,edges,nmc=1000,seed=24713,correct_edge=True,smooth=True):
    """Return hydrogen density, its uncertainty, emissivity and stage arrays."""
    inner=edges[:-1];outer=edges[1:]
    area=np.pi*(outer**2-inner**2)
    factor=count_to_hydrogen_emission_measure()
    emission=profile*factor*area;sigma=error*factor*area
    correction=edge_correction(inner,outer,emission) if correct_edge else np.zeros_like(emission)
    volume=projection_volumes(inner*ARCMIN_KPC*KPC_CM,outer*ARCMIN_KPC*KPC_CM)
    # The source uses NumPy's legacy MT19937 normal stream. The seed controls
    # uncertainty integration only; no hidden atmosphere or synthetic data exist.
    rng=np.random.RandomState(seed)
    draws=sigma[:,None]*rng.randn(len(profile),nmc)+emission[:,None]
    rhs=draws*(1-correction[:,None])
    samples=np.linalg.solve(volume,rhs)
    deviation=np.std(samples,axis=1);raw=np.median(samples,axis=1)
    emissivity=median_smooth(raw) if smooth else raw
    density=np.sign(emissivity)*np.sqrt(np.abs(emissivity))
    if smooth:density=median_smooth(density)
    uncertainty=.5/np.sqrt(np.abs(emissivity))*deviation
    return density,uncertainty,emissivity,{'volume':volume,'rhs':rhs,'samples':samples,'emission':emission,'correction':correction,'raw_emissivity':raw,'emissivity_std':deviation}

def reconstruct(data,nmc=1000,seed=24713):
    measured=background_subtraction(data)
    density,error,emissivity,stages=onion_peeling(measured['profile'],measured['error'],data['radial_edges'],nmc,seed)
    # v2 reports two physical representations of one density field. The
    # upstream separately smoothed q is retained only as historical diagnostics.
    if np.any(density<0):raise ValueError('Physical hydrogen density must be nonnegative')
    return {'reconstruction':density[None],'emissivity':density[None]**2},dict(measured=measured,density_uncertainty=error,historical_smoothed_emissivity=emissivity,**stages)
