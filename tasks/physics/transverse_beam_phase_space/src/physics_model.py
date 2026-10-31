"""Explicit covariance inversion, paired-mode normalization and4D density."""
import numpy as np
from .preprocessing import COORDS
VOXEL_EDGES=np.linspace(-4.,4.,36)
AXIS=(VOXEL_EDGES[:-1]+VOXEL_EDGES[1:])/2

def fit_covariance(moments,matrices):
    ax=[];ay=[];axy=[];bx=[];by=[];bxy=[]
    for M,(xx,yy,uu) in zip(matrices,moments):
        ax.append([M[0,0]**2,M[0,1]**2,2*M[0,0]*M[0,1]])
        ay.append([M[2,2]**2,M[2,3]**2,2*M[2,2]*M[2,3]])
        axy.append([M[0,0]*M[2,2],M[0,1]*M[2,2],M[0,0]*M[2,3],M[0,1]*M[2,3]])
        bx.append(xx);by.append(yy);bxy.append(.5*(xx+yy)-uu)
    xx=np.linalg.lstsq(ax,bx,rcond=None)[0];yy=np.linalg.lstsq(ay,by,rcond=None)[0];xy=np.linalg.lstsq(axy,bxy,rcond=None)[0]
    a,b,c=xx;d,e,f=yy;g,h,i,j=xy
    return np.array([[a,c,g,i],[c,b,h,j],[g,h,d,f],[i,j,f,e]])

def normalization(S):
    J=np.kron(np.eye(2),np.array([[0.,1.],[-1.,0.]]))
    _,vectors=np.linalg.eig(S@J)
    for k in [0,2]:
        value=(vectors[:,k].conj()@J@vectors[:,k]).imag
        if value>0:vectors[:,[k,k+1]]=vectors[:,[k+1,k]]
        vectors[:,k:k+2]*=np.sqrt(2/abs(value))
    V=np.empty((4,4))
    for k in [0,2]:V[:,k]=vectors[:,k].real;V[:,k+1]=(1j*vectors[:,k]).real
    N=np.linalg.inv(V);modal=N@S@N.T
    emittance=np.repeat([(modal[k,k]+modal[k+1,k+1])/2 for k in [0,2]],2)
    return np.linalg.inv(V@np.diag(np.sqrt(emittance)))

def grid_points():
    return np.stack([a.ravel() for a in np.meshgrid(*[AXIS]*4,indexing='ij')],axis=-1)

def projected_coordinates(z,V,matrices):
    x=z@V.T;out=[]
    for M in matrices:
        u=x@M.T
        out.extend([u[:,0],u[:,2],np.sum(u*(np.array([1.,0,-1.,0])/np.sqrt(2)),axis=1)])
    return np.asarray(out)

def probability(h,projected,prior):
    p=np.ones(len(prior))
    for value,position in zip(h,projected):
        p*=np.interp(position,COORDS,value,left=0,right=0)
    return p*prior

def emittances(covariance):
    """Geometric projected/intrinsic RMS emittances, in mm*mrad."""
    S=np.asarray(covariance)
    J=np.kron(np.eye(2),np.array([[0.,1.],[-1.,0.]]))
    tr=np.trace((S@J)@(S@J))
    discriminant=tr**2-16*np.linalg.det(S)
    root=np.sqrt(max(0.,discriminant))
    return dict(eps_x=float(np.sqrt(np.linalg.det(S[:2,:2]))),
                eps_y=float(np.sqrt(np.linalg.det(S[2:,2:]))),
                eps_1=float(.5*np.sqrt(-tr+root)),
                eps_2=float(.5*np.sqrt(-tr-root)))

