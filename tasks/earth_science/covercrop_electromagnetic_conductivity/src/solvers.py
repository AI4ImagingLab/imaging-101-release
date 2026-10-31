"""Local EMagPy bounded L2 extraction. Original EMagPy authors, GPL-3.0."""
import numpy as np
from scipy.optimize import minimize
from .physics_model import forward,jacobian
ALPHA=.07
def objective(p,observation,interfaces,spacings,alpha=ALPHA):
    """Square root of mean squared data error plus alpha*roughness/7."""
    r=forward(p,interfaces,spacings)-observation
    return np.sqrt(np.sum(r*r)/len(observation)+alpha*np.sum(np.diff(p)**2)/len(p))
def reconstruct(data,alpha=ALPHA):
    """Return (30,7) conductivity and per-station optimizer diagnostics."""
    outputs=[];diagnostics=[]
    for obs in data['apparent_conductivity']:
        r=minimize(objective,np.full(7,20.),args=(obs,data['layer_interfaces'],data['coil_spacing'],alpha),method='L-BFGS-B',bounds=[(0,300)]*7,options={'ftol':1e-12,'gtol':1e-8,'maxiter':1000})
        outputs.append(r.x);diagnostics.append({'success':bool(r.success),'status':int(r.status),'objective':float(r.fun),'iterations':int(r.nit)})
    return np.asarray(outputs),diagnostics
def quadratic_solution(data,alpha=ALPHA):
    """Unbounded minimizer of the same quadratic inside the square root."""
    J=jacobian(data['layer_interfaces'],data['coil_spacing']);D=np.diff(np.eye(7),axis=0)
    return np.linalg.solve(J.T@J/6+alpha*D.T@D/7,J.T@data['apparent_conductivity'].T/6).T
