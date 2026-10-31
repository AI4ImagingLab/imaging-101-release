"""Local RBF renderer, directional tensor acquisition and matched adjoint.

RBF and tensor weights extracted from MIT flyxdm; see NOTICE.
"""
import numpy as np
from scipy.sparse import csr_matrix
from .cuda_runtime import project
from .preprocessing import weights_and_directions

def radial_basis(support,size=8,scale=.16):
    m,n=support.shape;x,y=np.meshgrid(np.linspace(-1,1,m),np.linspace(-1,1,n),indexing='ij')
    xx,yy=np.meshgrid(np.linspace(-1,1,size),np.linspace(-1,1,size),indexing='ij');locations=np.array([xx.ravel(),yy.ravel()]).T
    b=np.zeros((m*n,len(locations)+1))
    for column,p in enumerate(locations):
        v=np.exp(-.5*((x-p[0])**2+(y-p[1])**2)/(2*scale*scale)).ravel();v[v<np.exp(-.5*(5/(2*scale)))]=0;b[:,column]=v*support.ravel()
    b[:,-1]=support.ravel();return csr_matrix(b)

def render(basis,coefficients,shape):
    return np.array([basis.dot(c).reshape(shape) for c in coefficients])

class TensorAcquisition:
    def __init__(self,data,size=8,scale=.16):
        self.data=data;self.shape=data['support'].shape;self.basis=radial_basis(data['support'],size,scale);self.count=self.basis.shape[1]
        self.weights,self.sampler=weights_and_directions(data)
        density=project(data['support'][None],data['angles'])[0];self.mask=density>0
        self.inverse=np.zeros_like(density);self.inverse[self.mask]=1/density[self.mask]
        projected=self.mean_project(self.basis.T.toarray().reshape(self.count,*self.shape)).reshape(self.count,-1).T.astype(float)
        self.projected_basis=projected
        self.jacobian=np.concatenate([projected*(self.weights*self.sampler[c]).reshape(-1,1) for c in range(6)],axis=1)
    def mean_project(self,images):
        p=project(images,self.data['angles']);p[:,self.mask]*=self.inverse[self.mask];p[:,~self.mask]=0;return p
    def forward(self,coefficients):
        image=render(self.basis,coefficients,self.shape);p=self.mean_project(image)
        return self.weights*np.sum(self.sampler*p,axis=0)
    def adjoint(self,residual):
        return (self.jacobian.T@residual.ravel()).reshape(6,self.count)
    def objective(self,flat):
        residual=self.forward(flat.reshape(6,self.count))-self.weights*self.data['strain_measurements']
        return np.sum(residual*residual),(2*self.adjoint(residual)).ravel()
    def field_prediction(self,field):
        return np.sum(self.sampler*self.mean_project(field),axis=0)
