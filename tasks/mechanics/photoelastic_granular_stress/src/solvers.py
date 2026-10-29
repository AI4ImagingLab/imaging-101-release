"""Constrained published pixel-SSE inverse with exact static equilibrium.
The original pepe optimizer's balance argument-order and integer-calibration
issues are explicitly corrected. This is not unchanged optimizer reproduction.
"""
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from .preprocessing import prepare,disk_mask,FSIGMA,CENTER,RADIUS,EFFECTIVE_PIXELS_PER_METRE
from .physics_model import stress_basis,forward,principal_difference

def constraint_matrix(beta):
    A=np.zeros((3,2*len(beta)));A[0,::2]=np.cos(beta);A[0,1::2]=np.sin(beta)
    A[1,::2]=-np.sin(beta);A[1,1::2]=np.cos(beta);A[2,1::2]=1
    return A

def decode(q):return np.hypot(q[::2],q[1::2]),np.arctan2(q[1::2],q[::2])

def objective(q,intensity,beta,fsigma=FSIGMA,center=CENTER,radius=RADIUS):
    f,a=decode(q);basis=stress_basis(beta,intensity.shape,alpha=a,center=center,radius=radius)
    residual=(forward(f,basis,fsigma)-intensity)*disk_mask(intensity.shape,center,.96*radius)
    return float(np.sum(residual**2))

def objective_gradient(q,intensity,beta,fsigma=FSIGMA,center=CENTER,radius=RADIUS):
    h=1e-6;eye=np.eye(len(q))*h
    return np.array([(objective(q+d,intensity,beta,fsigma,center,radius)-objective(q-d,intensity,beta,fsigma,center,radius))/(2*h) for d in eye])

def normbound(q):return 1-q[::2]**2-q[1::2]**2

def normjac(q):
    J=np.zeros((len(q)//2,len(q)))
    for i in range(len(q)//2):J[i,2*i:2*i+2]=-2*q[2*i:2*i+2]
    return J

def fit_forces(intensity,beta,fsigma=FSIGMA,initial=.04,balanced=True,center=CENTER,radius=RADIUS):
    q0=np.tile([initial,0.],len(beta));A=constraint_matrix(beta)
    B=np.zeros((2*len(beta),2*len(beta)));mu=np.tan(np.deg2rad(85))
    for j in range(len(beta)):
        B[2*j,2*j]=mu;B[2*j,2*j+1]=1;B[2*j+1,2*j]=mu;B[2*j+1,2*j+1]=-1
    constraints=[LinearConstraint(B,np.zeros(2*len(beta)),np.full(2*len(beta),np.inf)),{'type':'ineq','fun':normbound,'jac':normjac}]
    if balanced:constraints.insert(0,LinearConstraint(A,np.zeros(3),np.zeros(3)))
    args=(intensity,beta,fsigma,center,radius)
    result=minimize(objective,q0,args=args,jac=objective_gradient,method='SLSQP',bounds=[(1e-6,1),(-1,1)]*len(beta),constraints=constraints,options={'ftol':1e-9,'maxiter':500})
    if not result.success:raise RuntimeError(result.message)
    f,a=decode(result.x)
    return f,a,{'objective':float(result.fun),'iterations':result.nit,'cartesian':result.x,'constraint_residual':A@result.x}

def reconstruct(data,fsigma=FSIGMA,initial=.04,balanced=True,center=CENTER,radius=RADIUS):
    intensity=prepare(data['rgb']);forces,alpha,info=fit_forces(intensity,data['contact_angles'],fsigma,initial,balanced,center,radius)
    stress=principal_difference(forces,stress_basis(data['contact_angles'],intensity.shape,alpha=alpha,center=center,radius=radius))
    return {'reconstruction':stress[None], 'forces':forces[None], 'contact_directions':alpha[None]},info
