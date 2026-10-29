"""Original L-BFGS-B settings with explicitly corrected matched Jacobian."""
import numpy as np
from scipy.optimize import minimize
from .physics_model import TensorAcquisition,render

def reconstruct(data,maxiter=None,size=8,scale=.16):
    model=TensorAcquisition(data,size,scale);history=[]
    def objective(flat):
        loss,gradient=model.objective(flat);history.append([float(loss),float(np.max(abs(gradient)))]);return loss,gradient
    result=minimize(objective,np.zeros(6*model.count),tol=1e-8,method='L-BFGS-B',jac=True,options={'maxiter':6*model.count if maxiter is None else maxiter,'disp':False})
    coefficients=result.x.reshape(6,model.count);field=render(model.basis,coefficients,model.shape)
    return field,dict(model=model,coefficients=coefficients,history=np.array(history),iterations=int(result.nit),success=bool(result.success),message=str(result.message),objective=float(result.fun))
