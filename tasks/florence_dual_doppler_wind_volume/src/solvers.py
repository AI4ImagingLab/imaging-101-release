import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .physics_model import objective_gradient,forward
def reconstruct(data,mass_weight=256.,prior_weight=1e-4,tolerance=1e-9,max_iterations=2000):
    """Explicit local objective with general L-BFGS-B; return winds,prediction,state.
    Zero initialization, bounds±100m/s,history10,factr1,maxls40.
    """
    shape=(3,)+data['radial_velocity'].shape[1:];initial=np.zeros(np.prod(shape))
    x,value,state=fmin_l_bfgs_b(objective_gradient,initial,args=(data,mass_weight,prior_weight),bounds=[(-100.,100.)]*initial.size,maxiter=max_iterations,pgtol=tolerance,factr=1.,maxls=40,m=10)
    if state['warnflag']!=0:raise RuntimeError(state['task'])
    winds=x.reshape(shape);return winds,forward(winds,data),{'objective':float(value),'iterations':int(state['nit']),'projected_gradient_max':float(np.max(abs(state['grad'])))}
