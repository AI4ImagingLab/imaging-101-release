import numpy as np
from scipy.optimize import differential_evolution,least_squares
from .physics_model import LorenzMie,Instrument
from .preprocessing import regions

def model_at(parameters,coordinates):
    model=LorenzMie(coordinates=coordinates)
    model.instrument=Instrument(wavelength=.447,magnification=.048,n_m=1.34,noise=.05)
    model.particle.properties=dict(zip(('x_p','y_p','z_p','a_p','n_p'),parameters))
    return model.hologram()

def reconstruct(normalized,robust=True,seed=0,global_search=True):
    results=[]; diagnostics=[]
    for crop,coords,valid in regions(normalized):
        data=crop.ravel(); active=np.flatnonzero(valid)
        rng=np.random.RandomState(seed); subset=rng.choice(active,max(100,int(.02*len(active))),replace=False)
        center=(coords[0].mean(),coords[1].mean())
        def start(q):return np.array([*center,*q])
        if global_search:
            opt=differential_evolution(lambda q:np.sum(((model_at(start(q),coords[:,subset])-data[subset])/.05)**2),[(100.,350.),(.5,2.),(1.35,1.8)],seed=seed,popsize=10,maxiter=200,workers=1,polish=True)
            p0=start(opt.x)
        else:p0=np.array([*center,230.,1.,1.5])
        fit=least_squares(lambda p:(model_at(p,coords[:,valid])-data[valid])/.05,p0,bounds=([coords[0].min(),coords[1].min(),100.,.5,1.35],[coords[0].max(),coords[1].max(),350.,2.,1.8]),method='trf',loss='cauchy' if robust else 'linear',x_scale='jac',ftol=1e-9,xtol=1e-9,gtol=1e-9,max_nfev=300)
        results.append(fit.x); diagnostics.append(dict(cost=float(2*fit.cost/valid.sum()),nfev=fit.nfev,success=bool(fit.success)))
    return np.array(results)[None],diagnostics
