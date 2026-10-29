"""Published direct least-squares inverse, including retrospective sampling."""
import numpy as np
from .preprocessing import differential_coefficients
from .physics_model import inverse

def reconstruct(data,pair_count=4096,ordering='acquisition',seed=20261020):
    """Return batched64x64xchannels signed float64 reconstruction.

    Standard CLI uses all4096 actual complementary pairs and all2048 channels.
    Partial-pair experiments zero missing Walsh coefficients, evaluating the
    published minimum-norm inverse; they generate no synthetic observation.
    """
    if not isinstance(pair_count,(int,np.integer)) or not 1<=pair_count<=4096:raise ValueError('pair_count must be1..4096')
    coeff=differential_coefficients(data['measurements'],data['pattern_id'])
    acquired=data['pattern_id'][::2]//2
    if ordering=='acquisition':keep=acquired[:pair_count]
    elif ordering=='sequency':keep=np.arange(pair_count)
    elif ordering in ('random','random_dc'):
        positions=np.random.default_rng(seed).permutation(4096)
        if ordering=='random_dc':
            dc=np.flatnonzero(acquired==0)[0]
            positions=np.r_[dc,positions[positions!=dc]]
        keep=acquired[positions[:pair_count]]
    else:raise ValueError('Unknown sampling order')
    if pair_count<4096:
        mask=np.zeros(4096,dtype=bool);mask[keep]=True;coeff[~mask]=0
    return {'reconstruction':inverse(coeff.reshape(64,64,-1))[None]}
