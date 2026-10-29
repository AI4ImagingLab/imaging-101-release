"""Explicit recursive n-section and crossed temperature-density iteration."""
import numpy as np

DIAGNOSTICS={
    'NII':('N2',[(5,4)],[(4,3)]),
    'OII':('O2',[(3,1)],[(2,1)]),
    'OIII':('O3',[(5,4)],[(4,3)]),
    'SII':('S2',[(2,1)],[(3,1)]),
    'SIIblue':('S2',[(5,1),(4,1)],[(3,1),(2,1)]),
    'SIII':('S3',[(5,4)],[(3,2)]),
    'NeIII':('Ne3',[(4,1),(4,2)],[(2,1)])
}

def line_ratio(ions,name,temperature,density):
    ionname,numerator,denominator=DIAGNOSTICS[name]
    ion=ions[ionname];pop=ion.populations(temperature,density)
    def line(upper,lower):
        return pop[upper-1]*ion.A[upper-1,lower-1]*(ion.energy[upper-1]-ion.energy[lower-1])
    return sum(line(*pair) for pair in numerator)/sum(line(*pair) for pair in denominator)

def nsection(function,lower,upper,max_error=.001,n_cut=30,max_iter=20,iteration=0):
    if iteration>max_iter:return np.nan
    x=np.linspace(lower,upper,n_cut);y=np.abs(function(x));index=int(np.argmin(y))
    if y[index]<max_error:return x[index]
    index=max(1,min(n_cut-2,index))
    return nsection(function,x[index-1],x[index+1],max_error,n_cut,max_iter,iteration+1)

def invert_ratio(ions,name,observed,temperature=None,density=None):
    ion=ions[DIAGNOSTICS[name][0]]
    if temperature is None:
        lo,hi=ion.temperature[0],ion.temperature[-1]
        def residual(log_t):return (line_ratio(ions,name,10**log_t,density)-observed)/observed
    elif density is None:
        lo,hi=0.,8.
        def residual(log_n):return (line_ratio(ions,name,temperature,10**log_n)-observed)/observed
    else:raise ValueError('Exactly one of temperature/density must be unknown')
    return 10**nsection(residual,lo,hi)

def crossed_solution(ions,temperature_name,density_name,temperature_ratio,density_ratio):
    density=invert_ratio(ions,density_name,density_ratio,temperature=10000.)
    temperature=invert_ratio(ions,temperature_name,temperature_ratio,density=density)
    for _ in range(5):
        old_temperature,old_density=temperature,density
        density=invert_ratio(ions,density_name,density_ratio,temperature=old_temperature)
        # Preserve upstream Jacobi update: temperature uses OLD density.
        temperature=invert_ratio(ions,temperature_name,temperature_ratio,density=old_density)
        if abs(density-old_density)/density*100<=1 and abs(temperature-old_temperature)/temperature*100<=1:
            return np.array([temperature,density])
    return np.array([np.nan,np.nan])
