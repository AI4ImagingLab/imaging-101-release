"""Explicit conditional Gaussian-prior GLS/NNLS and tied deep-fault solve.

CSI source commit074ca028... selected author tutorial mathematics, GPL-3.0.
The general-purpose NNLS primitive is SciPy; all task-specific construction,
units, covariance regularization, prior shift, and equality mapping are local.
"""
import time
import numpy as np
from scipy import linalg
from scipy.optimize import nnls
from .physics_model import build_green_matrix, build_data_covariance, build_model_covariance
from .preprocessing import observation_vector


def assemble_problem(data, diagonal_noise=False, model_scale=1.):
    """Build fresh G3554x1305,d3554,Cd3554²,Cm1305² from visible arrays."""
    return (build_green_matrix(data), observation_vector(data),
            build_data_covariance(data, diagonal=diagonal_noise),
            build_model_covariance(data['fault_triangles'])*model_scale)


def tie_deep_parameters(green, covariance):
    """Eliminate deep indices1292..1296, append their column sum with variance100.

    Returns reduced G3554x1301,Cm1301²,mapping1305x1301. The new variance
    is the source's explicit100(mm/year)², not the sum of old variances.
    """
    keep=np.r_[np.arange(1292),np.arange(1297,1305)]
    mapping=np.zeros((1305,1301))
    mapping[keep,np.arange(1300)]=1.
    mapping[1292:1297,-1]=1.
    reduced=np.column_stack((green[:,keep],np.sum(green[:,1292:1297],axis=1)))
    cm=np.zeros((1301,1301));cm[:-1,:-1]=covariance[np.ix_(keep,keep)];cm[-1,-1]=100.
    return reduced,cm,mapping


def objective_gradient(model, green, observations, inverse_data_covariance,
                       inverse_model_covariance, prior):
    """Half data Mahalanobis plus half shifted Gaussian-prior quadratic."""
    residual=green@model-observations
    difference=model-prior
    value=.5*(residual@inverse_data_covariance@residual+difference@inverse_model_covariance@difference)
    gradient=green.T@(inverse_data_covariance@residual)+inverse_model_covariance@difference
    return float(value),gradient


def solve_system(green, observations, data_covariance, model_covariance,
                 method='tied_nnls'):
    """Solve one native comparison; return full1305model and numeric diagnostics.

    GLS prior0; NNLS shifted prior -100mm/year for all5datum parameters and
    zero elsewhere. NNLS enforces m-prior>=0. The final method ties5deep
    rates exactly. Neither optimization method estimates elastic geometry.
    """
    if method not in ('gls','nnls','tied_nnls'):
        raise ValueError('Unknown comparison method')
    start=time.perf_counter()
    g=green;cm=model_covariance;mapping=None
    if method=='tied_nnls':
        g,cm,mapping=tie_deep_parameters(g,cm)
    inverse_cd=np.linalg.inv(data_covariance)
    inverse_cm=linalg.inv(cm)
    prior=np.zeros(g.shape[1])
    if method=='gls':
        h=g.T@inverse_cd@g+inverse_cm
        rhs=g.T@inverse_cd@observations
        model=linalg.inv(h)@rhs
        residual_norm=None
    else:
        datum=np.arange(1295,1300) if mapping is not None else np.arange(1300,1305)
        prior[datum]=-100.
        left=np.linalg.cholesky(inverse_cd)
        right=np.linalg.cholesky(inverse_cm)
        augmented=np.vstack((left.T@g,right.T))
        rhs=np.r_[left.T@(observations-g@prior),np.zeros(g.shape[1])]
        increment,residual_norm=nnls(augmented,rhs)
        model=increment+prior
    value,gradient=objective_gradient(model,g,observations,inverse_cd,inverse_cm,prior)
    raw_residual=observations-g@model
    weighted_rms=float(np.sqrt(raw_residual@inverse_cd@raw_residual/len(observations)))
    diagnostics={'method':method,'runtime_seconds':time.perf_counter()-start,
                 'objective':value,'weighted_observation_rms':weighted_rms,
                 'observation_rms_mm_per_year':float(np.sqrt(np.mean(raw_residual**2))),
                 'gradient':gradient,'reduced_model':model.copy(),'prior':prior,
                 'residual_norm':residual_norm}
    if method!='gls':
        increment=model-prior
        diagnostics['projected_gradient_inf']=float(np.max(abs(increment-np.maximum(increment-gradient,0.))))
    return (mapping@model if mapping is not None else model),diagnostics


def output_arrays(model):
    """Complete batch1 output:1299mm/year slips,1formal kPa/year coefficient,5datums."""
    model=np.asarray(model)
    if model.shape!=(1305,) or not np.isfinite(model).all():
        raise ValueError('Expected finite1305parameter solution')
    return {'slip_rate':model[:1299][None], 'basin_coefficient':model[1299:1300][None],
            'datum_offset':model[1300:][None]}


def reconstruct(data, method='tied_nnls', diagonal_noise=False, model_scale=1.):
    """Fresh full geometry→operators→inverse, returning model1305 and diagnostics."""
    start=time.perf_counter()
    result,details=solve_system(*assemble_problem(data,diagonal_noise,model_scale),method=method)
    details['total_runtime_seconds']=time.perf_counter()-start
    return result,details
