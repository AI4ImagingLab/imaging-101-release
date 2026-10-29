"""Explicit factorized helioseismic acquisition, adjoint and SOLA targets.

Adapted from SOLApack's pseudo-Kronecker kernel formula (see LICENSE).
"""
import numpy as np
from .preprocessing import quadrature

class RotationOperator:
    """Native weighted kernel contractions; never materializes a dense K."""
    def __init__(self,data):
        self.f1=data['radial_kernel_1'];self.f2=data['radial_kernel_2']
        self.g1=data['angular_kernel_1'];self.g2=data['angular_kernel_2']
        self.inl=data['radial_index'];self.ilm=data['angular_index']
        error=np.abs(data['splitting_error']);self.inv_error=np.ones_like(error)
        np.divide(1,error,out=self.inv_error,where=error!=0)
        self.nr,self.nnl=self.f1.shape;self.nt,self.nlm=self.g1.shape
        self.nm=len(self.inl);self.np=self.nr*self.nt
        # Identify native contiguous multiplets, then merge equal angular blocks.
        cuts=np.r_[0,np.flatnonzero((np.diff(self.inl)!=0)|(np.diff(self.ilm)!=1))+1,self.nm]
        chunks=[(int(a),int(b)) for a,b in zip(cuts[:-1],cuts[1:])]
        blocks=[];pos=0
        while pos<len(chunks):
            start,end=chunks[pos];first=int(self.inl[start]);angular=self.ilm[start:end];last=first;pos+=1
            while pos<len(chunks):
                aa,bb=chunks[pos]
                if int(self.inl[aa])!=last+1 or not np.array_equal(self.ilm[aa:bb],angular):break
                last+=1;end=bb;pos+=1
            blocks.append((start,end,first,last+1,int(angular[0]),int(angular[-1])+1))
        self.blocks=blocks
        self.weight,self.sqrt_weight,self.inv_sqrt_weight=quadrature(data)

    def forward(self,field,whiten=True):
        """Flattened theta-fast field nHz -> all measured coefficients nHz."""
        x=np.asarray(field).reshape(self.nr,self.nt).T
        a=x@self.f1;b=x@self.f2;y=np.empty(self.nm)
        for lo,hi,n0,n1,l0,l1 in self.blocks:
            y[lo:hi]=(self.g1[:,l0:l1].T@a[:,n0:n1]+self.g2[:,l0:l1].T@b[:,n0:n1]).ravel(order='F')
        return y*self.inv_error if whiten else y

    def adjoint(self,coefficients,whiten=True):
        """Exact algebraic transpose of forward, with matching whitening."""
        y=np.asarray(coefficients)*self.inv_error if whiten else np.asarray(coefficients)
        a=np.zeros((self.nt,self.nnl));b=np.zeros_like(a)
        for lo,hi,n0,n1,l0,l1 in self.blocks:
            block=y[lo:hi].reshape(l1-l0,n1-n0,order='F')
            a[:,n0:n1]+=self.g1[:,l0:l1]@block
            b[:,n0:n1]+=self.g2[:,l0:l1]@block
        return (a@self.f1.T+b@self.f2.T).ravel(order='F')

def gaussian_targets(mesh,radial_widths,target_indices,angular_width=.1):
    """Equator-symmetrized normalized native target columns on full mesh."""
    radius=mesh[:,0,None];theta=mesh[:,1,None]
    index=np.asarray(target_indices,dtype=int);center=mesh[index]
    dr=radial_widths[index//51][None,:]
    radial=((radius-center[None,:,0])/dr)**2
    a=((theta-center[None,:,1])*radius/angular_width)**2
    b=((theta-(np.pi-center[None,:,1]))*radius/angular_width)**2
    target=np.exp(-radial-a)+np.exp(-radial-b)
    w=mesh[:,0]*mesh[:,2]
    return target/(w@target)[None,:]

def initial_target(mesh):
    """Native unscaled starting target r0=.7, theta0=.7, dr=.04, dt=.1."""
    r=mesh[:,0];t=mesh[:,1]
    radial=((r-.7)/.04)**2
    p=np.exp(-radial-((t-.7)*r/.1)**2)+np.exp(-radial-((t-(np.pi-.7))*r/.1)**2)
    return p/np.dot(mesh[:,0]*mesh[:,2],p)

class ConstrainedOperator:
    """A = W^-1/2 K_whiten.T H[:,1:], with exact Householder constraint."""
    def __init__(self,operator):
        self.op=operator
        integrals=operator.forward(np.ones(operator.np))
        norm=np.linalg.norm(integrals);signed=-np.copysign(norm,integrals[0])
        self.tau=1/(norm*(norm+abs(integrals[0])))
        self.v=integrals.copy();self.v[0]-=signed
        self.signed=signed;self.alpha=1/signed
        self.q=-self.tau*self.v[0]*self.alpha*self.v;self.q[0]+=self.alpha
        self.constraint_kernel=operator.inv_sqrt_weight*operator.adjoint(self.q)

    def householder(self,x):
        return x-self.tau*self.v*np.dot(self.v,x)

    def forward(self,x):
        full=np.r_[0.,x];full=self.householder(full)
        return self.op.inv_sqrt_weight*self.op.adjoint(full)

    def adjoint(self,x):
        full=self.op.forward(self.op.inv_sqrt_weight*x)
        return self.householder(full)[1:]
