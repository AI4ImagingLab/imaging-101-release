"""Diagnostic perturbations only; never regenerate the bound author observations.

add_displacement_noise uses NumPy PCG64(default_rng) at an explicit seed and
adds independent zero-mean Gaussian sensor displacement noise in pixels, with
standard deviation multiplier*supplied MC random sigma. Positions/calibration
and every node remain unchanged. This is a sensitivity diagnostic, not new truth.
"""
import numpy as np
def add_displacement_noise(observations,multiplier=1.,seed=123):
    if isinstance(multiplier,bool) or not np.isfinite(multiplier) or multiplier<0:raise ValueError('Invalid noise multiplier')
    result={k:np.array(v,copy=True) for k,v in observations.items()}
    rng=np.random.default_rng(seed)
    result['displacement']+=multiplier*result['displacement_sigma']*rng.standard_normal(result['displacement'].shape)
    return result

def conditional_density_samples(diagnostics,gradient_sigma,n_draw=2048,seed=731):
    """Gaussian linear covariance diagnostic, kg/m³ deviations, no truth.

    Draw independent nodal gradient errors in kg/m4 and boundary errors in
    kg/m3 under the selected fixed conditional covariance model, then solve
    the unchanged WLS normal equations for their density deviations. The seed
    is explicit PCG64. This validates algebra, not physical uncertainty coverage.
    """
    from scipy.sparse.linalg import splu
    if isinstance(n_draw,bool) or not isinstance(n_draw,int) or n_draw<2:raise ValueError('Need at least2 draws')
    rng=np.random.default_rng(seed);d=diagnostics;sg=np.asarray(gradient_sigma).ravel()
    if sg.size!=d['B'].shape[1] or np.any(sg<0) or not np.isfinite(sg).all():raise ValueError('Invalid gradient sigma')
    gnoise=sg[:,None]*rng.standard_normal((len(sg),n_draw))
    bnoise=1e-10*rng.standard_normal((d['C'].shape[1],n_draw))
    rhs=d['B']@gnoise+d['C']@bnoise
    target=d['D'].T@(d['weights'][:,None]*rhs)
    return splu(d['normal'].tocsc()).solve(target)
