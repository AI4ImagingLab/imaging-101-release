import numpy as np
from scipy.optimize import fsolve,minimize,differential_evolution

def initial_energy(raw,temperature=303.15):
    x=raw['membership'].astype(float);exchange=raw['exchanges'];capacity=raw['capacity'];t=raw['time_s'];rfu=raw['measurements']/raw['control'][:,None]
    z=x*exchange[None,:]/capacity[:,None]
    with np.errstate(divide='ignore',invalid='ignore'):norm=z/z.sum(axis=0)[None,:]
    f=norm.T@rfu;boundaries=np.r_[0,np.flatnonzero(np.any(np.diff(x,axis=1)!=0,axis=0))+1,x.shape[1]]
    lower=np.log(2)/(t[t>0].max()*50);upper=50*np.log(2)/t[t>0].min();rates=np.full(x.shape[1],np.nan);counter=0
    for start,stop in zip(boundaries[:-1],boundaries[1:]):
        d=f[start]
        if np.all(np.isnan(d)):continue
        k1=fsolve(lambda k:(1-np.exp(-k*t[2]))-d[2],.5)[0]
        k2=fsolve(lambda k:(1-(.5*np.exp(-k1*t[-2])+.5*np.exp(-k*t[-2])))-d[-2],.05)[0]
        order=sorted(range(3),key=lambda j:f'pyhdx_par_{counter+j}');counter+=3
        def unpack(v):
            p=np.empty(3);p[order]=v;return p
        def loss(v):
            r,a,b=unpack(v)
            with np.errstate(over='ignore',invalid='ignore'):pred=-r*np.exp(-a*t)-(1-r)*np.exp(-b*t)+1
            return .5*np.sum((pred-d)**2)
        p0=np.array([.5,k1,k2]);res=minimize(loss,p0[order],method='Powell',tol=1e-9);p=unpack(res.x);bounds=[(0,1),(lower,upper),(lower,upper)]
        if not all(lo<=v<=hi for v,(lo,hi) in zip(p,bounds)) or not np.all(np.isfinite(p)) or 2*res.fun>.2:
            res=differential_evolution(loss,[bounds[j] for j in order],strategy='rand1bin',popsize=40,mutation=(.423,1.053),recombination=.95,polish=False,init='latinhypercube',seed=np.random.RandomState(43));p=unpack(res.x)
        r,a,b=p;initial=a if r>.5 else b
        loghalf=fsolve(lambda logt:.5-r*np.exp(-a*np.exp(logt))-(1-r)*np.exp(-b*np.exp(logt)),np.log(initial))
        rates[start:stop]=np.log(2)/np.exp(loghalf.item())
    factor=raw['intrinsic_rates']/rates-1;factor=np.maximum(factor,0)
    with np.errstate(divide='ignore',invalid='ignore'):energy=np.log(factor)*8.31446261815324*temperature
    bad=~np.isfinite(energy);energy[bad]=np.interp(np.flatnonzero(bad),np.flatnonzero(~bad),energy[~bad]);return energy
