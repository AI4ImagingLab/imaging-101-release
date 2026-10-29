"""Full eight-segment processing and wavelength-domain thin-screen arc inversion."""
import time,numpy as np
from .preprocessing import load_observations,fill_gaps,stitch
from .physics_model import physical_kappa

def reconstruct(path='data/raw_data.npz'):
    start=time.perf_counter();segments=load_observations(path);total=None;results=[];diagnostics=[]
    for d in segments:
        d.trim_edges();d.dyn=fill_gaps(d.dyn);d.correct_dyn()
        d.calc_sspec();d.scale_dyn()
        d.fit_arc(etamin=2,etamax=400,delmax=1,numsteps=1000,nsmooth=7,startbin=2)
        results.append(float(physical_kappa(d.betaeta)));diagnostics.append(d)
        total=d if total is None else stitch(total,d)
    total.dyn=fill_gaps(total.dyn,method='mean');total.correct_dyn();total.calc_sspec(lamsteps=True)
    total.fit_arc(etamin=2,etamax=400,delmax=.3,numsteps=1000,nsmooth=5,cutmid=3)
    results.append(float(physical_kappa(total.betaeta)));diagnostics.append(total)
    return np.array(results,dtype=np.float64)[None],diagnostics,{'seconds':time.perf_counter()-start,'native_segment_count':len(segments)}
