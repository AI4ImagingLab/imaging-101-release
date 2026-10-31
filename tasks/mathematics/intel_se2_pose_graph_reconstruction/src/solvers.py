import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from .physics_model import residual_jacobians,weighted_error,wrap
from .preprocessing import initialize

def normal_equations(poses,data):
    """Assemble exact sparse J^T Omega J and J^T Omega e; no robust kernel."""
    e,a,b=residual_jacobians(poses,data);o=data['information'];ids=data['edge_indices']
    nodes=np.concatenate((ids[:,0,None]*3+np.arange(3),ids[:,1,None]*3+np.arange(3)),axis=1)
    j=np.concatenate((a,b),axis=2);h=np.einsum('nai,nab,nbj->nij',j,o,j);g=np.einsum('nai,nab,nb->ni',j,o,e)
    n=poses.size;gradient=np.zeros(n);np.add.at(gradient,nodes.ravel(),g.ravel())
    rows=np.broadcast_to(nodes[:,:,None],h.shape);cols=np.broadcast_to(nodes[:,None,:],h.shape)
    matrix=coo_matrix((h.ravel(),(rows.ravel(),cols.ravel())),shape=(n,n)).tocsc()
    return matrix,gradient

def reconstruct(data,iterations=20):
    """Original canonical Gauss-Newton schedule; world-additive free-node update."""
    poses=initialize(data);history=[weighted_error(poses,data)]
    for _ in range(iterations):
        h,g=normal_equations(poses,data);step=spsolve(h[3:,3:],-g[3:],permc_spec='COLAMD')
        if not np.all(np.isfinite(step)): raise ArithmeticError('Nonfinite normal-equation solution')
        poses[1:]+=step.reshape(-1,3);poses[:,2]=wrap(poses[:,2]);history.append(weighted_error(poses,data))
    return poses,np.asarray(history)
