"""Explicit orthographic projection and native metric-upgrade equations.

Rigid-helper stages adapted from Alessio Del Bue and Lourdes Agapito,
18/08/2009, GPLv2; see LICENSE and NOTICE. No domain library is imported.
"""
import numpy as np
from scipy.linalg import sqrtm

def symmetric_products(a,b):
    """Six coefficients of a@C@b for symmetric3x3 C, order00,01,02,11,12,22."""
    return np.array([a[0]*b[0],a[0]*b[1]+a[1]*b[0],a[0]*b[2]+a[2]*b[0],a[1]*b[1],a[1]*b[2]+a[2]*b[1],a[2]*b[2]])

def metric_system(motion):
    """Return A[3F,6],b[3F] for both unit row norms and orthogonality."""
    x=motion[::2];y=motion[1::2];F=len(x)
    A=np.array([symmetric_products(a,a) for a in x]+[symmetric_products(a,a) for a in y]+[symmetric_products(a,b) for a,b in zip(x,y)])
    return A,np.concatenate((np.ones(2*F),np.zeros(F)))

def metric_upgrade(motion,shape):
    """Native closed-form metric upgrade plus joint row-orthogonality step.

    Returns upgraded M[2F,3],S[3,P],diagnostics. The last step is exactly
    G@sqrtm(lstsq(num,den)), with den framewise pseudoinverse(Mframe.T).
    """
    A,b=metric_system(motion);v=np.linalg.lstsq(A,b,rcond=None)[0]
    C=np.array([[v[0],v[1],v[2]],[v[1],v[3],v[4]],[v[2],v[4],v[5]]])
    U,s,_=np.linalg.svd(C);G=U@np.diag(np.sqrt(s))
    num=motion@G
    den=np.concatenate([np.linalg.pinv(num[i:i+2].T) for i in range(0,len(num),2)],axis=0)
    correction=np.linalg.lstsq(num,den,rcond=None)[0]
    root=sqrtm(correction)
    if np.iscomplexobj(root):
        if np.max(np.abs(root.imag))>1e-12:raise ValueError('Non-real native metric square root')
        root=root.real
    Q=G@root
    return motion@Q,np.linalg.inv(Q)@shape,dict(A=A,b=b,C=C,G=G,num=num,den=den,correction=correction,Q=Q)

def forward(shape,cameras,translations):
    """Project shape[P,3] with cameras[F,2,3],translations[F,2] ->[F,P,2]."""
    return np.einsum('fkc,pc->fpk',cameras,shape)+translations[:,None,:]

def joint_alignment(shape,cameras,reference_shape):
    """One global orthogonal/reflection gauge, jointly applied; no scaling."""
    u,_,vh=np.linalg.svd(shape.T@reference_shape);Q=u@vh
    return shape@Q,cameras@Q,Q
