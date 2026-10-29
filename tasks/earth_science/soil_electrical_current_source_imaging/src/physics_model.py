"""Explicit calibrated electrical-current superposition and adjoint.
BSD-3-Clause; adapted from Copyright2018 ICSD Developers.
"""
import numpy as np
def forward(current_fractions,green_matrix):
    """(N,) dimensionless source fractions -> (M,) transfer resistance ohm."""
    return green_matrix@np.asarray(current_fractions).ravel()
def adjoint(residual,green_matrix):return green_matrix.T@residual
def regularizer(source_coordinates):
    """Original selected connected graph, including its missing boundary edges.
    Each row is one unscaled nearest-neighbor difference. Source order is x,y.
    """
    nx=np.unique(np.round(source_coordinates[:,0],3)).size;ny=np.unique(np.round(source_coordinates[:,1],3)).size;n=nx*ny;nodes=np.arange(n).reshape(ny,nx);rows=[]
    for y in range(ny):
        for x in range(nx):
            if x+1<nx:
                row=np.zeros(n);row[nodes[y,x]]=-1;row[nodes[y,x+1]]=1;rows.append(row)
                if y+1<ny:
                    row=np.zeros(n);row[nodes[y,x]]=-1;row[nodes[y+1,x]]=1;rows.append(row)
    return np.array(rows)
def weighted_system(data,regularization_weight=1.,conservation_weight=1000.):
    """Return stacked Aw,bw,R,w with explicit data/current/smoothness rows."""
    from .preprocessing import observation_weights
    G=data['green_matrix'];b=data['observations'];R=regularizer(data['source_coordinates']);w=observation_weights(b)
    A=np.vstack([G,np.ones((1,G.shape[1])),R]);target=np.concatenate([b,[1.],np.zeros(R.shape[0])]);weights=np.concatenate([w,[conservation_weight],np.full(R.shape[0],regularization_weight)])
    return weights[:,None]*A,weights*target,R,w
def objective_gradient(x,A,b):
    """Half squared stacked residual and exact gradient; no hidden normalization."""
    r=A@x-b
    return .5*float(r@r),A.T@r
