# SPDX-License-Identifier: LGPL-3.0
# Adapted from MTtime, Copyright2020 Lawrence Livermore National Security LLC.
"""Explicit Herrmann fundamental-fault radiation mixing in NED coordinates."""
import numpy as np

def design_matrix(greens, azimuth, degree=6):
    """(station,10,time) cm/(1e20 dyne cm) -> (station*3*time,degree).

    Fundamental order ZSS,ZDS,ZDD,ZEX,RSS,RDS,RDD,REX,TSS,TDS.
    Coefficients order NN,NE,ND,EE,ED,DD. Decimal constants intentionally
    preserve upstream precision; see validation.md for isotropic leakage.
    """
    if degree not in (5,6):
        raise ValueError('degree must be5 or6')
    n,_,nt=greens.shape
    G=np.zeros((n,3,nt,degree),dtype=np.float64)
    for i,a in enumerate(np.asarray(azimuth)*np.pi/180):
        g=np.asarray(greens[i],dtype=np.float64)
        c,s,c2,s2=np.cos(a),np.sin(a),np.cos(2*a),np.sin(2*a)
        for component,offset in [(0,0),(1,4)]:
            G[i,component,:,1]=g[offset]*s2
            G[i,component,:,2]=g[offset+1]*c
            G[i,component,:,4]=g[offset+1]*s
            if degree==6:
                G[i,component,:,0]=.5*g[offset]*c2-.166667*g[offset+2]+.33333*g[offset+3]
                G[i,component,:,3]=-.5*g[offset]*c2-.166667*g[offset+2]+.33333*g[offset+3]
                G[i,component,:,5]=.33333*g[offset+2]+.33333*g[offset+3]
            else:
                G[i,component,:,0]=.5*g[offset]*c2-.5*g[offset+2]
                G[i,component,:,3]=-.5*g[offset]*c2-.5*g[offset+2]
        G[i,2,:,0]=.5*g[8]*s2
        G[i,2,:,3]=-.5*g[8]*s2
        G[i,2,:,1]=-g[8]*c2
        G[i,2,:,2]=g[9]*s
        G[i,2,:,4]=-g[9]*c
    return G.reshape(-1,degree)

def coefficients_to_tensor(a):
    """Native dimensionless coefficients -> symmetric3x3 NED moment in Nm."""
    dd=a[5] if len(a)==6 else -(a[0]+a[3])
    return 1e13*np.array([[a[0],a[1],a[2]],[a[1],a[3],a[4]],[a[2],a[4],dd]])

def tensor_to_coefficients(tensor):
    """3x3 Nm -> six dimensionless radiation coefficients."""
    m=np.asarray(tensor)/1e13
    return m[[0,0,0,1,1,2],[0,1,2,1,2,2]]

def forward(greens,azimuth,tensor):
    """Predict(8,3,150) displacement in cm."""
    return (design_matrix(greens,azimuth)@tensor_to_coefficients(tensor)).reshape(greens.shape[0],3,-1)

def adjoint(matrix,residual):
    """Euclidean adjoint of the explicit linear radiation operator."""
    return matrix.T@np.asarray(residual).ravel()
