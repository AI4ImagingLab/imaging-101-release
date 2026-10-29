"""Explicit source2026 residual-noise/robust-fit and full stretch-posterior pipeline.

SteParSyn mathematical conventions, copyright2015–2026 Tabernero, Marfil,
Gonzalez Hernandez and Montes, BSD-2-Clause. Goodman-Weare red/blue transition
conventions follow emcee3.1.6 (MIT). No upstream/domain solver is imported.
"""
import numpy as np
from scipy.optimize import curve_fit
from .preprocessing import mask_flux

INITIAL=np.array([5.777,4.44,0.,3.8],dtype=np.float64)
LOWER=np.array([3.5,0.,-2.,.2],dtype=np.float64)
UPPER=np.array([7.,5.5,1.,100.],dtype=np.float64)

def window_error(synthetic,observed,uncertainty,wavelength,resolution=.2,threshold=3.,variant='source'):
    """Original source residual heuristic, with an explicit aligned diagnostic.

    The source ranks zip(wavelength[mask],abs(full_residual)), whose indices
    differ after filtering. This known2026 source convention is deliberately
    identified, not silently corrected or attributed to the2022 paper equation.
    The input uncertainty is unused by this original residual-only estimator.
    """
    residual=synthetic-observed;median=np.median(residual);scatter=1.4826*np.median(np.abs(residual-median))
    mask=np.abs(residual-median)<=scatter*threshold;tentative=wavelength[mask];covered=np.zeros_like(tentative,dtype=bool);absolute=np.abs(residual)
    if variant not in ['source','aligned_diagnostic']:raise ValueError('Unknown residual ranking convention')
    ranking=absolute if variant=='source' else absolute[mask]
    selected=[];centers=[];widths=[];amplitudes=[]
    distance=(np.max(wavelength)-np.mean(wavelength))/(len(wavelength)-1)
    for wave,value in sorted(zip(tentative,ranking),key=lambda p:p[1],reverse=True):
        if wave in tentative[covered]:continue
        selected.append(wave);covered|=(tentative>=wave-resolution)&(tentative<=wave+resolution)
    for center in selected:
        pick=(wavelength>center-resolution)&(wavelength<center+resolution);weighted=np.abs(residual[pick]-median);waves=wavelength[pick];peak=residual[pick]
        centroid=np.sum(weighted*waves)/np.sum(weighted)
        width=np.sqrt(distance**2+np.sum(weighted*(waves-centroid)**2)/np.sum(weighted));amplitude=np.max(np.abs(peak))
        if abs(amplitude)>=threshold*scatter:centers.append(centroid);widths.append(width);amplitudes.append(amplitude)
    return np.asarray(centers),np.asarray(widths),np.asarray(amplitudes),float(scatter)

def estimate_error_map(model,parameters,variant='source'):
    """Recompute source Gaussian residual-map over every window, from observations."""
    raw=model.raw;spectrum=model.forward_fidelity(parameters);centers=[];widths=[];amplitudes=[];scatter=[];per_window=[]
    for lo,hi in raw['ranges']:
        center=.5*(lo+hi);width=.5*(hi-lo);pick=abs(raw['wavelength']-center)<=width
        result=window_error(spectrum[pick],raw['flux'][pick],raw['uncertainty'][pick],raw['wavelength'][pick],variant=variant)
        centers.extend(result[0]);widths.extend(result[1]);amplitudes.extend(result[2]);scatter.append(result[3]);per_window.append(result)
    centers=np.asarray(centers);widths=np.asarray(widths);amplitudes=np.asarray(amplitudes);error=np.zeros(len(raw['wavelength']))
    for index,(lo,hi) in enumerate(raw['ranges']):
        center=.5*(lo+hi);width=.5*(hi-lo);pick=abs(raw['wavelength']-center)<=width;wave=raw['wavelength'][pick]
        segment=error[pick].copy()
        for c,s,a in zip(centers,widths,amplitudes):segment+=a*np.exp(-.5*((c-wave)/s)**2.)
        error[pick]=np.sqrt(segment**2.+scatter[index]**2.)
    # Source computes an error floor into an unused variable. Preserve the
    # actually returned unfloored map; an initial-photon-error floor is NOT used.
    masked=mask_flux(raw['wavelength'],raw['flux'],error,raw['masks'])[2]
    return error,masked,dict(centers=centers,widths=widths,amplitudes=amplitudes,scatter=np.asarray(scatter),per_window=per_window)

def dual_curve_fit(model,masked_wave,masked_flux,masked_error,tolerance=.001,initial=None):
    """Original pair of robust Huber TRF/LSMR fits, relative and absolute covariance."""
    def prediction(wave,*parameters):
        try:return model.forward_fidelity(np.asarray(parameters),masked=True)
        except ValueError:
            result=np.ones(len(masked_flux));result[0]=1000.;return result
    start=INITIAL.copy() if initial is None else np.asarray(initial,dtype=np.float64).copy()
    arguments=dict(p0=start,sigma=masked_error,method='trf',bounds=(LOWER,UPPER),xtol=tolerance,tr_solver='lsmr',loss='huber')
    parameters,covariance=curve_fit(prediction,masked_wave,masked_flux,absolute_sigma=False,**arguments)
    absolute_parameters,absolute_covariance=curve_fit(prediction,masked_wave,masked_flux,absolute_sigma=True,**arguments)
    factor=float(np.mean(np.sqrt(np.diag(covariance))/np.sqrt(np.diag(absolute_covariance))))
    return parameters,covariance,factor

def prepare_likelihood(model):
    """Complete original preprocessing, both dual fits and runtime error-map estimate."""
    raw=model.raw;masked_wave,masked_flux,initial_error,velocity,ranked_velocity=mask_flux(raw['wavelength'],raw['flux'],raw['uncertainty'],raw['masks'])
    first,first_covariance,first_factor=dual_curve_fit(model,masked_wave,masked_flux,initial_error)
    rescaled_error=first_factor*initial_error
    full_noise,masked_noise,map_details=estimate_error_map(model,first)
    # Original wide_errors invokes turn_params, which mutates configuration
    # values to the first fit. Its next get_params therefore starts here.
    second,second_covariance,second_factor=dual_curve_fit(model,masked_wave,masked_flux,rescaled_error,initial=first)
    errors=second_factor*masked_noise;parameter_error=np.sqrt(np.diag(second_covariance));width=np.maximum(.01*np.ones(4),parameter_error)
    return dict(parameters=second,covariance=second_covariance,parameter_error=parameter_error,walker_width=width,errors=errors,masked_wavelength=masked_wave,masked_flux=masked_flux,masked_input_error=rescaled_error,full_error_map=full_noise,masked_error_map=masked_noise,error_map_details=map_details,first_parameters=first,first_covariance=first_covariance,first_factor=first_factor,second_factor=second_factor,masked_velocity=velocity,ranked_velocity=ranked_velocity)

def log_probability(parameters,model,likelihood):
    """Uniform inclusive bounds and original Gaussian source likelihood, no2π constant."""
    p=np.asarray(parameters)
    if np.any(p>UPPER) or np.any(p<LOWER):return -np.inf
    try:prediction=model.forward(p,masked=True)
    except ValueError:return -np.inf
    if not np.isfinite(prediction).all():return -np.inf
    variance=likelihood['errors']**2;residual=likelihood['masked_flux']-prediction
    return float(-.5*np.sum(residual*residual/variance+np.log(variance)))

def log_probability_batch(parameters,model,likelihood):
    """Ordered deterministic evaluation only; sampler RNG/updates remain sequential."""
    parameters=np.asarray(parameters);values=np.full(len(parameters),-np.inf);valid=np.all((parameters<=UPPER)&(parameters>=LOWER),axis=1)
    if not hasattr(model,'forward_batch'):
        return np.array([log_probability(p,model,likelihood) for p in parameters])
    # Domain/hull eligibility is checked by the same local physical operator.
    try:
        predictions=model.forward_batch(parameters[valid],masked=True)
    except ValueError:return np.array([log_probability(p,model,likelihood) for p in parameters])
    variance=likelihood['errors']**2
    for index,prediction in zip(np.flatnonzero(valid),predictions):
        if np.isfinite(prediction).all():
            residual=likelihood['masked_flux']-prediction
            values[index]=-.5*np.sum(residual*residual/variance+np.log(variance))
    return values

def initialize_walkers(initial,width,model,likelihood,seed=20261028):
    """Original sixteen uniform starting walkers, with its source fallback convention."""
    random=np.random.RandomState(seed);positions=[]
    for dimension in range(4):
        for repeat in range(4):positions.append(initial+2.*width*(random.random_sample(4)-.5))
    probability=np.array([log_probability(p,model,likelihood) for p in positions]);good=[positions[j] for j in np.flatnonzero(np.isfinite(probability))]
    if not good:raise ValueError('No valid initial walkers')
    for j in range(16):
        if not np.isfinite(probability[j]):positions[j]=good[random.randint(len(good)-1)]
    return np.asarray(positions)

def stretch_posterior(model,likelihood,burn=100,production=1000,seed=20261028,progress=None):
    """Explicit a=2 red-blue ensemble, exactly16walkers and full100+1000 protocol."""
    # Original sampler clones the RNG BEFORE set_walkers draws from global RNG.
    random=np.random.RandomState(seed)
    positions=initialize_walkers(likelihood['parameters'],likelihood['walker_width'],model,likelihood,seed)
    probability=log_probability_batch(positions,model,likelihood);indices=np.arange(16);chain=np.empty((16,production,4));logp=np.empty((16,production));acceptance=np.zeros(16,dtype=np.int64)
    for phase,steps in [('burn',burn),('production',production)]:
        if phase=='production':probability=log_probability_batch(positions,model,likelihood)
        for step in range(steps):
            random.choice(1,p=np.array([1.]));groups=indices%2;random.shuffle(groups);accepted=np.zeros(16,dtype=bool)
            for split in range(2):
                subset=groups==split;sets=[positions[groups==j] for j in range(2)];current=sets[split];complement=np.concatenate(sets[:split]+sets[split+1:],axis=0)
                stretch=((2.-1.)*random.rand(len(current))+1)**2./2.;factors=(4.-1.)*np.log(stretch);partners=random.randint(len(complement),size=(len(current),))
                proposed=complement[partners]-(complement[partners]-current)*stretch[:,None];proposed_probability=log_probability_batch(proposed,model,likelihood)
                for j,f,new in zip(indices[subset],factors,proposed_probability):
                    with np.errstate(invalid='ignore'):difference=f+new-probability[j]
                    if difference>np.log(random.rand()):accepted[j]=True
                update=subset&accepted;positions[update]=proposed[accepted[subset]];probability[update]=proposed_probability[accepted[subset]]
            if phase=='production':chain[:,step]=positions;logp[:,step]=probability;acceptance+=accepted
            if progress is not None and (step+1)%10==0:progress(phase,step+1,float(np.max(probability)))
    return chain,logp,acceptance/float(production)

def reconstruct(model,seed=20261028,progress=None):
    """Complete selected inverse, including runtime inferred noise and full posterior."""
    likelihood=prepare_likelihood(model);chain,logp,acceptance=stretch_posterior(model,likelihood,seed=seed,progress=progress)
    selected=logp==np.max(logp);best=np.array(chain[selected,:][0]);spectrum=model.forward(best);physical=best.copy();physical[0]*=1000.
    return dict(reconstruction=physical[None],spectrum=spectrum[None]),dict(chain=chain,log_probability=logp,acceptance_fraction=acceptance,likelihood=likelihood,best_internal=best)
