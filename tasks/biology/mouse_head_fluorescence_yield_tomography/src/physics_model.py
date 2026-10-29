"""Explicit P1 diffusion/fluorescence FEM, adapted from BSD3 NIRFAST.

The qualified Robin term integrates all topology-defined exterior triangles.
The source continuum triple-P1 inexact adjoint update is retained separately
from the exact derivative of the nodal-product fluorescence source.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

MUAX=.009
MUAM=.006
KAPPAX=0.251953
KAPPAM=0.26062
RI=1.33
ETA=.1

def tetra_geometry(nodes,elements):
    """Return element volumes [mm^3] and P1 gradients [mm^-1]."""
    p=nodes[elements]
    affine=np.concatenate((np.ones((*p.shape[:2],1)),p),axis=2)
    gradients=np.linalg.inv(affine)[:,1:,:]
    volumes=np.abs(np.linalg.det(p[:,1:]-p[:,:1]))/6
    if not np.all(volumes>0):raise ValueError('Degenerate tetrahedron')
    return volumes,gradients

def assemble_local(elements,values,n):
    """Sum the local square matrices into an explicit global sparse matrix."""
    return coo_matrix((values.ravel(),(np.broadcast_to(elements[:,:,None],values.shape).ravel(),np.broadcast_to(elements[:,None,:],values.shape).ravel())),shape=(n,n)).tocsc()

def exterior_faces(elements):
    faces=np.concatenate([np.sort(elements[:,f],axis=1) for f in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))])
    faces,counts=np.unique(faces,axis=0,return_counts=True)
    if np.any(counts>2):raise ValueError('Nonmanifold tetrahedron mesh')
    return faces[counts==1]

def refractive_robin(ri=RI):
    """Dimensionless source Fresnel partial-current boundary coefficient."""
    r0=((ri-1)/(ri+1))**2
    theta=np.arcsin(1/ri)
    a=(2/(1-r0)-1+abs(np.cos(theta))**3)/(1-abs(np.cos(theta))**2)
    return 1/(2*a)

def assemble_fem(nodes,elements,mua,kappa,ksi=None):
    """CW weak diffusion operator and nodal-product mass matrix, units mm."""
    n=len(nodes);vol,grad=tetra_geometry(nodes,elements)
    mass=vol[:,None,None]*(np.ones((4,4))+np.eye(4))[None]/20
    stiffness=kappa*vol[:,None,None]*np.einsum('eki,ekj->eij',grad,grad)
    matrix=assemble_local(elements,stiffness+mua*mass,n)
    faces=exterior_faces(elements);p=nodes[faces]
    area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/2
    if ksi is None:ksi=refractive_robin()
    robin=area[:,None,None]*ksi*(np.ones((3,3))+np.eye(3))[None]/12
    matrix=matrix+assemble_local(faces,robin,n)
    return matrix.tocsc(),assemble_local(elements,mass,n),vol

def continuum_prejacobian(elements,vol,phix,adjoint,links):
    """Original continuum triple-P1 integral; not the exact nodal-chain Jacobian."""
    n=len(phix);rows=[]
    for source,detector in links:
        x=phix[elements,source];a=np.conj(adjoint[elements,detector])
        sx=x.sum(axis=1,keepdims=True);sa=a.sum(axis=1,keepdims=True)
        local=vol[:,None]/120*(sx*sa+(x*a).sum(axis=1,keepdims=True)+x*sa+a*sx+2*x*a)
        row=np.bincount(elements.ravel(),weights=local.real.ravel(),minlength=n)+1j*np.bincount(elements.ravel(),weights=local.imag.ravel(),minlength=n)
        rows.append(row)
    return np.asarray(rows)

def forward_fluorescence(gamma,phix,mass,emission_lu,detector,links):
    beta=np.where(gamma==0,1e-20,gamma)
    field=emission_lu.solve(np.asarray(mass@(beta[:,None]*phix)))
    measured=detector@field
    values=measured[links[:,1],links[:,0]]
    return np.abs(values),field
