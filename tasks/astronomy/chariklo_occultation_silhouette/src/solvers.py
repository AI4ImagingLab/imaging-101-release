"""Original Monte Carlo chi-square ellipse search and negative-chord filtering."""
import numpy as np
from .preprocessing import radial_uncertainty
from .physics_model import contact_chi2,negative_margin,ellipse_contour

def monte_carlo_fit(data,seed=24713,loop=100000,number_chi=20000,dchi_min=10.):
    rng=np.random.RandomState(seed);accepted=[];objectives=[];batches=0
    center=np.array([5.416,-3.747,138.,.093,90.]);width=np.array([10.,20.,50.,.20,90.])
    sigma=radial_uncertainty(data)
    while sum(len(x) for x in objectives)<number_chi:
        trials=np.column_stack([c+w*(2*rng.random_sample(loop)-1) for c,w in zip(center,width)])
        trials[:,3]=np.clip(trials[:,3],0,1)
        chi=contact_chi2(trials,data['limb_points'],sigma)
        # The source compares with each batch's minimum, not the running minimum.
        keep=chi<chi.min()+dchi_min;accepted.append(trials[keep]);objectives.append(chi[keep]);batches+=1
    trials=np.concatenate(accepted);chi=np.concatenate(objectives)
    clear=np.array([negative_margin(p,data['negative_path'])>0 for p in trials])
    if not np.any(clear):raise ValueError('no ellipse remains compatible with the negative chord')
    valid=np.flatnonzero(clear);best=valid[np.argmin(chi[clear])]
    return trials[best],{'parameters':trials,'chi2':chi,'negative_clear':clear,'best_index':best,'batches':batches}

def reconstruct(data,seed=24713,loop=100000,number_chi=20000):
    best,diagnostics=monte_carlo_fit(data,seed,loop,number_chi)
    return {'parameters':best[None],'reconstruction':ellipse_contour(best,data['contour_angles'])[None]},diagnostics
