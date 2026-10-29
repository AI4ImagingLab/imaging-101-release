"""Shear-spring assembly, generalized eigenpairs, explicit residuals/Jacobians."""
import numpy as np
from scipy.linalg import eigh
from .preprocessing import MASS,NOMINAL_STIFFNESS,normalize_modes
def influence_matrices():
    d=np.eye(4)
    for j in range(1,4):d[j,j-1]=-1
    return NOMINAL_STIFFNESS*np.einsum('ji,jk->jik',d,d)
def stiffness_matrix(alpha):return np.einsum('j,jik->ik',1+np.asarray(alpha),influence_matrices())
def eigenmodes(alpha,q):
    values,vectors=eigh(stiffness_matrix(alpha),np.eye(4)*MASS)
    vectors=vectors[:,:2];vectors/=vectors[q,np.arange(2)]
    return values[:2],vectors
def residual_jacobian(x,frequency,measured,form=3):
    """Return residual and analytic Jacobian: form1 MAC,2 eigenshape,3 dynamics."""
    y,q=normalize_modes(measured);lam=(2*np.pi*np.asarray(frequency))**2
    K=stiffness_matrix(x[:4]);M=np.eye(4)*MASS;influences=influence_matrices()
    if form==3:
        u=np.vstack((y,np.asarray(x[4:])[None]));r=[];J=np.zeros((8,6))
        for m in range(2):
            operator=K-lam[m]*M;r.extend(operator@u[:,m]);J[4*m:4*m+4,:4]=np.array([v@u[:,m] for v in influences]).T;J[4*m:4*m+4,4+m]=operator[:,3]
        return np.asarray(r),J
    values,u=eigenmodes(x,q);r=[];jac=[]
    for m in range(2):
        v=u[:,m];mass=v@M@v
        dl=np.array([v@d@v/mass for d in influences]);keep=np.delete(np.arange(4),q[m]);B=(K-values[m]*M)[np.ix_(keep,keep)]
        dv=np.zeros((4,4))
        for j,d in enumerate(influences):dv[keep,j]=np.linalg.solve(B,(dl[j]*M@v-d@v)[keep])
        r.append((lam[m]-values[m])/lam[m]);jac.append(-dl/lam[m])
        if form==1:
            a=y[:,m];b=v[:3];rootmac=abs(a@b)/np.linalg.norm(a)/np.linalg.norm(b)
            r.append(1/rootmac-1);jac.append(-1/rootmac*(a/(a@b)-b/(b@b))@dv[:3])
        elif form==2:
            selected=np.delete(np.arange(3),q[m]);r.extend(y[selected,m]-v[selected]);jac.extend(-dv[selected])
        else:raise ValueError('form must be1,2,3')
    return np.asarray(r),np.asarray(jac)
