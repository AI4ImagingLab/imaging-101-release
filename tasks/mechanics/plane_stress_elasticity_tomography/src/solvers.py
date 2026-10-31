"""Local weighted Gauss-Newton, explicit spatial priors and source line search.

Adapted mathematical stages: Danny Smyl, OpenQSEI, 10.2.2018, Aalto University.
See NOTICE and validation for the independently qualified source discrepancies.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix,diags
from .preprocessing import Acquisition,element_incenters,observation_weights
from .physics_model import PlaneStressModel

MINIMUM_MODULUS=0.1
UPPER_PENALTY_THRESHOLD=300.0
TIKHONOV_WEIGHT=1e-5
MAXIMUM_ITERATION_INDEX=100
RELATIVE_COST_STOP=1e-6
QHULL_OPTIONS='Qt Qbb Qc'

@dataclass
class InverseContext:
    model: PlaneStressModel
    centers: np.ndarray
    weights: np.ndarray
    initial_modulus: np.ndarray
    gradient_x: object
    gradient_y: object
    dual_triangles: np.ndarray
    alpha: float
    beta: float
    gaussian_precision: np.ndarray

@dataclass
class ReconstructionResult:
    reconstruction: np.ndarray
    modulus_history: np.ndarray
    objective_history: np.ndarray
    step_sizes: np.ndarray
    directions: np.ndarray
    predicted_displacement: np.ndarray
    runtime_seconds: float
    prior: str
    stop_reason: str

def gradient_operators(centers: np.ndarray):
    """Affine gradient on the fixed source-compatible incenter triangulation."""
    triangles=Delaunay(centers,qhull_options=QHULL_OPTIONS).simplices
    xy=centers[triangles];x=xy[:,:,0];y=xy[:,:,1]
    determinant=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0])
    gx=np.column_stack([y[:,1]-y[:,2],y[:,2]-y[:,0],y[:,0]-y[:,1]])/determinant[:,None]
    gy=np.column_stack([x[:,2]-x[:,1],x[:,0]-x[:,2],x[:,1]-x[:,0]])/determinant[:,None]
    rows=np.repeat(np.arange(len(triangles)),3);shape=(len(triangles),len(centers))
    Rx=coo_matrix((gx.ravel(),(rows,triangles.ravel())),shape=shape).tocsr()
    Ry=coo_matrix((gy.ravel(),(rows,triangles.ravel())),shape=shape).tocsr()
    return Rx,Ry,triangles

def homogeneous_initializer(model: PlaneStressModel, measured: np.ndarray) -> np.ndarray:
    """Original candidate grid, correct sliced index, exact homogeneous scaling.

    K(c*1)=c*K(1), including diagonal replacement; one unit forward solve
    therefore evaluates the same 191 retained candidate predictions exactly
    algebraically. The candidate subset is source indices10..200 inclusive.
    """
    unit_displacement=model.forward(np.ones(model.element_count))
    candidates=5+1.25*np.arange(200,dtype=np.float64)
    errors=np.linalg.norm(unit_displacement[None,:]/candidates[:,None]-
                          measured.reshape(1,-1),axis=1)
    selected=9+int(np.argmin(errors[9:]))
    return np.full(model.element_count,candidates[selected],dtype=np.float64)

def gaussian_prior_precision(centers: np.ndarray, mean: float) -> np.ndarray:
    """Source correlated Gaussian covariance (modulus squared), explicitly built."""
    variance=((4*mean-50)/6)**2
    correlation_length=1.0
    b=np.sqrt(-correlation_length**2/(2*np.log(.01)))
    difference=centers[:,None,:]-centers[None,:,:]
    squared_distance=np.sum(difference*difference,axis=2)
    covariance=.999*variance*np.exp(-squared_distance/(2*b*b))
    covariance.flat[::len(centers)+1]+=.001*variance
    return np.linalg.inv(covariance)

def prepare_inverse(acquisition: Acquisition) -> InverseContext:
    model=PlaneStressModel(acquisition)
    centers=element_incenters(acquisition.nodes,acquisition.triangles)
    initial=homogeneous_initializer(model,acquisition.displacement)
    Rx,Ry,dual=gradient_operators(centers)
    maximum_gradient=2*initial[0]/.5
    return InverseContext(model,centers,observation_weights(acquisition.displacement),
                          initial,Rx,Ry,dual,float(-np.log(.5)/maximum_gradient),
                          float(.001*maximum_gradient**2),
                          gaussian_prior_precision(centers,float(initial[0])))

def tv_value_gradient_hessian(modulus: np.ndarray, Rx, Ry, beta: float,
                              need_hessian: bool=True):
    """Unweighted-face stabilized TV, its exact gradient and Hessian."""
    x=Rx@modulus;y=Ry@modulus;s=np.sqrt(x*x+y*y+beta)
    value=float(s.sum());gradient=Rx.T@(x/s)+Ry.T@(y/s)
    hessian=None
    if need_hessian:
        inverse_cube=1/s**3
        xx=diags(1/s-x*x*inverse_cube);yy=diags(1/s-y*y*inverse_cube)
        xy=diags(-x*y*inverse_cube)
        hessian=(Rx.T@xx@Rx+Rx.T@xy@Ry+Ry.T@xy@Rx+Ry.T@yy@Ry).toarray()
    return value,np.asarray(gradient),hessian

def prior_terms(modulus: np.ndarray, context: InverseContext, prior: str,
                alpha_multiplier: float=1., need_hessian: bool=True):
    """Return regularizer value, gradient and optional dense Hessian."""
    if prior=='tv':
        value,gradient,hessian=tv_value_gradient_hessian(modulus,context.gradient_x,
                                         context.gradient_y,context.beta,need_hessian)
        scale=context.alpha*alpha_multiplier
        return scale*value,scale*gradient,None if hessian is None else scale*hessian
    if prior=='smooth':
        difference=modulus-context.initial_modulus[0]
        gradient=context.gaussian_precision@difference
        return float(.5*difference@gradient),gradient,context.gaussian_precision if need_hessian else None
    if prior=='tikhonov':
        return float(.5*TIKHONOV_WEIGHT*(modulus@modulus)),TIKHONOV_WEIGHT*modulus,TIKHONOV_WEIGHT*np.eye(len(modulus)) if need_hessian else None
    raise ValueError('Prior must be tv, smooth or tikhonov')

class Objective:
    def __init__(self, context: InverseContext, prior: str='tv',alpha_multiplier: float=1.):
        self.context=context;self.prior=prior;self.alpha_multiplier=float(alpha_multiplier)
        if not np.isfinite(self.alpha_multiplier) or self.alpha_multiplier<=0:
            raise ValueError('TV multiplier must be finite and positive')
        self.measured=context.model.acquisition.displacement.reshape(-1)
        self.sqrt_weights=np.sqrt(context.weights)
        self.lower_coefficient=0.;self.upper_coefficient=0.
        self.initial_objective,_=self.value(context.initial_modulus)
        Q=1e-4*self.initial_objective
        self.lower_coefficient=Q/(.01*MINIMUM_MODULUS)**2
        self.upper_coefficient=Q/UPPER_PENALTY_THRESHOLD**2

    def barrier(self, modulus: np.ndarray):
        lower=np.minimum(modulus-MINIMUM_MODULUS,0)
        upper=np.maximum(modulus-UPPER_PENALTY_THRESHOLD,0)
        value=float(self.lower_coefficient*(lower@lower)+self.upper_coefficient*(upper@upper))
        gradient=2*self.lower_coefficient*lower+2*self.upper_coefficient*upper
        diagonal=2*self.lower_coefficient*(lower<0)+2*self.upper_coefficient*(upper>0)
        return value,gradient,diagonal

    def value(self, modulus: np.ndarray):
        prediction=self.context.model.forward(modulus)
        mismatch=self.sqrt_weights*(self.measured-prediction)
        regularizer=prior_terms(modulus,self.context,self.prior,self.alpha_multiplier,False)[0]
        value=float(.5*np.linalg.norm(mismatch)**2+regularizer+self.barrier(modulus)[0])
        return value,prediction

    def derivatives(self, modulus: np.ndarray):
        _,gradient,hessian=prior_terms(modulus,self.context,self.prior,self.alpha_multiplier,True)
        _,barrier_gradient,diagonal=self.barrier(modulus)
        return gradient+barrier_gradient,hessian+np.diag(diagonal)

def gauss_newton_direction(modulus: np.ndarray, prediction: np.ndarray,
                            jacobian: np.ndarray, objective: Objective) -> np.ndarray:
    gradient,hessian=objective.derivatives(modulus)
    weights=objective.context.weights
    left=jacobian.T@(weights[:,None]*jacobian)+hessian
    right=jacobian.T@(weights*(objective.measured-prediction))-gradient
    return np.linalg.solve(left,right)

def _unique_minimum(values: np.ndarray) -> int:
    indices=np.flatnonzero(values==np.min(values))
    if len(indices)!=1:
        raise FloatingPointError('Exact source line-search tie requires explicit qualification')
    return int(indices[0])

def uniform_line_search(modulus: np.ndarray, direction: np.ndarray,
                         objective: Objective, return_grids: bool=False):
    """The author's complete 100+100 point projected uniform search."""
    increment=1. if objective.prior=='tikhonov' else .02
    steps=np.empty(100);values=np.empty(100);step=1e-10
    for index in range(100):
        if index:step+=increment
        steps[index]=step
        trial=np.maximum(modulus+step*direction,MINIMUM_MODULUS)
        values[index]=objective.value(trial)[0]
    coarse=steps[_unique_minimum(values)]
    step=coarse-increment
    if step<0:step=1e-10
    refined_steps=np.empty(100);refined_values=np.empty(100)
    for index in range(100):
        step+=increment/(100/3)
        refined_steps[index]=step
        trial=np.maximum(modulus+step*direction,MINIMUM_MODULUS)
        refined_values[index]=objective.value(trial)[0]
    result=float(refined_steps[_unique_minimum(refined_values)])
    if return_grids:return result,steps,values,refined_steps,refined_values
    return result

def reconstruct(context: InverseContext, prior: str='tv',alpha_multiplier: float=1.,
                maximum_iteration_index: int=MAXIMUM_ITERATION_INDEX,
                verbose: bool=False) -> ReconstructionResult:
    """Execute complete chosen prior; default source cap is99 accepted updates."""
    if not isinstance(maximum_iteration_index,int) or not 2<=maximum_iteration_index<=100:
        raise ValueError('Iteration index must be an integer in[2,100]')
    start=time.perf_counter();objective=Objective(context,prior,alpha_multiplier)
    modulus=context.initial_modulus.copy();cost,prediction=objective.value(modulus)
    fields=[modulus.copy()];costs=[cost];steps=[];directions=[];reason='iteration_cap'
    for iteration in range(2,maximum_iteration_index+1):
        jacobian=context.model.jacobian(modulus)
        direction=gauss_newton_direction(modulus,prediction,jacobian,objective)
        step=uniform_line_search(modulus,direction,objective)
        modulus=np.maximum(modulus+step*direction,MINIMUM_MODULUS)
        cost,prediction=objective.value(modulus)
        fields.append(modulus.copy());costs.append(cost);steps.append(step);directions.append(direction)
        if verbose:print(f'{prior} iteration{iteration}: cost={cost:.12g}, step={step:.8g}',flush=True)
        relative=abs(cost-costs[-2])/costs[-2]
        if relative<RELATIVE_COST_STOP:reason='relative_cost';break
        if cost>costs[-2]:reason='cost_increase';break
    return ReconstructionResult(modulus,np.asarray(fields),np.asarray(costs),
             np.asarray(steps),np.asarray(directions),prediction,time.perf_counter()-start,prior,reason)
