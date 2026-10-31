"""Five explicit published inverse methods and identified diagnostic variants."""
import numpy as np
from scipy.optimize import minimize_scalar,fsolve
from scipy.interpolate import interp1d
from scipy.stats import linregress
from .physics_model import R,senum_yang,interval_integrals,ratio_objective,ratio_variance,temperature_integral

def linear_energy(temperature,rate,beta,method='Fr',power=1.92):
    out=[]
    for t,d in zip(temperature,rate):
        if method=='Fr':y=np.log(d);factor=R
        elif method=='OFW':y=np.log(beta);factor=R/1.052
        elif method=='KAS':y=np.log(beta)-np.log(t**power);factor=R
        else:raise ValueError('unknown linear method')
        fit=linregress(1/t,y)
        out.append([-factor*fit.slope,-factor*fit.stderr,fit.intercept])
    return np.asarray(out).T

def reported_error(energy,integral):
    """Original interpolated F diagnostic; not a calibrated confidence guarantee."""
    n=len(integral(energy));F=[161.4,19.,9.277,6.388,5.050,4.284,3.787,3.438,3.179,2.978,2.687][n-2]
    grid=np.linspace(1,energy+50,50)
    v0=ratio_variance(integral(energy))
    values=np.array([ratio_variance(integral(e)) for e in grid])/v0-F-1
    interp=interp1d(grid,values,kind='cubic',bounds_error=False,fill_value='extrapolate')
    roots=np.array([fsolve(interp,energy-150)[0],fsolve(interp,energy+150)[0]])
    return np.mean(np.abs(energy-roots))

def integral_energy(data,advanced=True,mode='time',with_errors=True):
    count=len(data['advanced_time'])-1 if advanced else len(data['temperature'])
    energies=[];errors=[]
    for j in range(count):
        if advanced:integral=lambda e:interval_integrals(e,j,data,mode)
        else:integral=lambda e:senum_yang(e,data['temperature'][j])/data['beta']
        result=minimize_scalar(lambda e:ratio_objective(integral(e)),bounds=(5,120),method='bounded',options={'xatol':1e-5,'maxiter':500})
        if not result.success:raise RuntimeError(result.message)
        energies.append(result.x)
        if with_errors:
            # Original temperature-branch uncertainty omits beta, unlike its fit.
            # Preserve and identify this diagnostic discrepancy rather than correcting silently.
            uncertainty=(lambda e:temperature_integral(e,*data['advanced_temperature'][j:j+2])) if advanced and mode=='temperature' else integral
            errors.append(reported_error(result.x,uncertainty))
    return np.array([energies,errors]) if with_errors else np.asarray(energies)

def reconstruct(data,comparisons=False):
    result={'reconstruction':integral_energy(data,with_errors=comparisons)}
    if comparisons:
        result['aVy']=result.pop('reconstruction')
        result['reconstruction']=result['aVy'][0][None]
        for name in ['Fr','OFW','KAS']:result[name]=linear_energy(data['temperature'],data['rate'],data['beta'],name)
        result['Vy']=integral_energy(data,advanced=False)
        result['aVy_temperature']=integral_energy(data,mode='temperature')
        result['KAS_classic']=linear_energy(data['temperature'],data['rate'],data['beta'],'KAS',2)
    else:result['reconstruction']=result['reconstruction'][None]
    return result
