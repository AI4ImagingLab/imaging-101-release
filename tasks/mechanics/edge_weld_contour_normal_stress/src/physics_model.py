"""Explicit fully integrated C3D8 linear elasticity in N, mm and MPa.

Mathematics extracted from the GPL3 pyCM contour method; finite-element assembly
is local. The qualified Gauss-to-node map is inverse shape interpolation in
native integration order, independently checked using prescribed exact fields.
"""
import numpy as np
from scipy.sparse import coo_matrix

NODE_SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)
GAUSS_SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]],dtype=float)
GAUSS=GAUSS_SIGNS/np.sqrt(3.)

def elastic_tensor(modulus=195600.,poisson=.294):
    if not np.isfinite([modulus,poisson]).all() or modulus<=0 or not -1<poisson<.5:raise ValueError('Invalid isotropic elasticity')
    mu=modulus/(2*(1+poisson));lam=modulus*poisson/((1+poisson)*(1-2*poisson))
    D=np.zeros((6,6));D[:3,:3]=lam;D[np.arange(3),np.arange(3)]+=2*mu;D[3:,3:]=mu*np.eye(3)
    return D

def shape_gradient(natural):
    result=np.empty((8,3))
    for axis in range(3):
        other=[k for k in range(3) if k!=axis]
        result[:,axis]=NODE_SIGNS[:,axis]*np.prod(1+NODE_SIGNS[:,other]*np.asarray(natural)[other],axis=1)/8
    return result

def strain_matrix(element_coordinates,natural):
    deriv=shape_gradient(natural)
    jac=np.einsum('eia,ib->eab',element_coordinates,deriv)
    det=np.linalg.det(jac)
    if not np.isfinite(det).all() or np.any(det<=0):raise ValueError('Nonpositive brick Jacobian')
    grad=np.einsum('ia,eab->eib',deriv,np.linalg.inv(jac))
    B=np.zeros((len(element_coordinates),6,24))
    B[:,0,0::3]=grad[:,:,0];B[:,1,1::3]=grad[:,:,1];B[:,2,2::3]=grad[:,:,2]
    B[:,3,0::3]=grad[:,:,1];B[:,3,1::3]=grad[:,:,0]
    B[:,4,0::3]=grad[:,:,2];B[:,4,2::3]=grad[:,:,0]
    B[:,5,1::3]=grad[:,:,2];B[:,5,2::3]=grad[:,:,1]
    return B,det

def element_dofs(elements):
    return (np.asarray(elements,dtype=np.int32)[:,:,None]*3+np.arange(3,dtype=np.int32)).reshape(-1,24)

def assemble_stiffness(nodes,elements,modulus=195600.,poisson=.294):
    coordinates=nodes[elements];D=elastic_tensor(modulus,poisson)
    local=np.zeros((len(elements),24,24))
    for natural in GAUSS:
        B,det=strain_matrix(coordinates,natural)
        local+=np.einsum('eai,ab,ebj,e->eij',B,D,B,det,optimize=True)
    dofs=element_dofs(elements)
    row=np.repeat(dofs,24,axis=1).ravel();col=np.tile(dofs,(1,24)).ravel()
    return coo_matrix((local.ravel(),(row,col)),shape=(len(nodes)*3,len(nodes)*3)).tocsc()

def integration_stress(nodes,elements,displacement,modulus=195600.,poisson=.294):
    D=elastic_tensor(modulus,poisson);ue=displacement[elements].reshape(-1,24);result=[]
    for natural in GAUSS:
        B,_=strain_matrix(nodes[elements],natural)
        strain=np.einsum('eai,ei->ea',B,ue)
        result.append(strain@D.T)
    return np.stack(result,axis=1)

def nodal_normal_stress(elements,gauss_stress,n_nodes):
    interpolation=np.prod(1+GAUSS[:,None,:]*NODE_SIGNS[None,:,:],axis=2)/8
    extrapolation=np.linalg.inv(interpolation)
    values=gauss_stress[:,:,2]@extrapolation.T
    ids=elements.ravel()
    counts=np.bincount(ids,minlength=n_nodes)
    if np.any(counts==0):raise ValueError('Unused FE node')
    return np.bincount(ids,weights=values.ravel(),minlength=n_nodes)/counts
