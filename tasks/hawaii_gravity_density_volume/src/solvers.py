"""Local weighted least squares and explicit Newton-CG inverse updates.

Adapted from MIT SimPEG v0.25.2 selected fixed-beta linear inversion.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg,LinearOperator
from .preprocessing import prepare_observations,project_trend
from .physics_model import gravity_matrix
def regularization_matrix():
    """Cell-volume smallness and three interior spatial differences.

    Uniform volume is 3000*3000*2500 m^3. The smallness coefficient is
    1/3000^2; derivative terms use physical metre spacings. Exterior normal
    derivatives are zero (natural boundary); zero-weight second derivatives
    of the original general regularizer contribute no rows here.
    """
    n=2048;rows=[];cols=[];values=[];row=0
    for i in range(n):rows.append(row);cols.append(i);values.append(50.);row+=1
    for axis,step,count,weight in [(0,1,16,50.),(1,16,16,50.),(2,256,8,60.)]:
        for z in range(8):
            for y in range(16):
                for x in range(16):
                    coord=(x,y,z)[axis]
                    if coord==count-1:continue
                    i=x+16*y+256*z
                    rows.extend([row,row]);cols.extend([i,i+step]);values.extend([-weight,weight]);row+=1
    return sparse.csr_matrix((values,(rows,cols)),shape=(row,n))
def objective_gradient(model,matrix,data,regularizer,beta=.001,sigma=5.):
    """Unhalved SimPEG L2 objective and gradient, density g/cm^3."""
    residual=matrix@model-data;penalty=regularizer@model
    fidelity=float(np.dot(residual,residual)/sigma**2);roughness=float(np.dot(penalty,penalty))
    gradient=2*(matrix.T@residual/sigma**2+beta*(regularizer.T@penalty))
    return fidelity+beta*roughness,gradient,fidelity,roughness
def newton_cg(matrix,data,regularizer,beta=.001,sigma=5.,max_iterations=10):
    """Original identity-preconditioned Newton-CG, with Armijo backtracking.

    CG solves each task-specific Hessian system to relative tolerance 1e-12,
    max2000; the original absolute gradient stopper is 1e3*eps=.01.
    The bound strictly quadratic problem converges after one full Newton step.
    """
    n=matrix.shape[1];model=np.zeros(n);history=[];steps=[]
    def hessian(vector):return 2*(matrix.T@(matrix@vector)/sigma**2+beta*(regularizer.T@(regularizer@vector)))
    operator=LinearOperator((n,n),matvec=hessian,dtype=np.float64)
    initial_objective=None;previous=None
    for iteration in range(max_iterations+1):
        value,gradient,fidelity,roughness=objective_gradient(model,matrix,data,regularizer,beta,sigma)
        if initial_objective is None:initial_objective=value
        norm=float(np.linalg.norm(gradient));history.append([value,fidelity,roughness,norm])
        if norm<=.01 or iteration==max_iterations:break
        if previous is not None and abs(previous-value)<=1e-12*(1+abs(initial_objective)):break
        direction,status=cg(operator,-gradient,rtol=1e-12,atol=0.,maxiter=2000)
        if status:raise RuntimeError('Hessian CG did not converge: '+str(status))
        step=1.;slope=float(np.dot(gradient,direction))
        for line_iteration in range(10):
            trial=model+step*direction
            new_value=objective_gradient(trial,matrix,data,regularizer,beta,sigma)[0]
            if new_value<=value+1e-4*step*slope:break
            step*=.5
        else:raise RuntimeError('Armijo line search failed')
        previous=value;model=trial;steps.append(step)
    return model,np.array(history),np.array(steps)
def reconstruct(data,beta=.001):
    """Return density (8,16,16), and actual acquisition/inversion diagnostics."""
    locations,anomaly,stages=prepare_observations(data)
    raw_matrix=gravity_matrix(locations);matrix=project_trend(raw_matrix,stages['trend']);regularizer=regularization_matrix()
    model,history,steps=newton_cg(matrix,anomaly,regularizer,beta=beta)
    return model.reshape(8,16,16),dict(locations=locations,anomaly=anomaly,kernel=raw_matrix,projected_kernel=matrix,regularizer=regularizer,history=history,steps=steps,prediction=matrix@model,**stages)
