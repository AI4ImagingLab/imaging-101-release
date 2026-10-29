"""Explicit P1 plane-stress FEM and the author's central-difference Jacobian.

Method provenance: Danny Smyl, OpenQSEI, 10.2.2018, Aalto University.
The associated SoftwareX article designates the source code MIT; see NOTICE.
No upstream package is imported. Geometry, material law, assembly, diagonal
replacement boundary convention, perturbation states and strain are explicit.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from .preprocessing import (Acquisition,POISSON_RATIO,THICKNESS,
                            BOUNDARY_MULTIPLIER)

DIFFERENCE_STEP=(np.finfo(np.float64).eps/2)**(1/3)

class PlaneStressModel:
    def __init__(self, acquisition: Acquisition, nu: float=POISSON_RATIO,
                 thickness: float=THICKNESS,
                 boundary_multiplier: float=BOUNDARY_MULTIPLIER):
        self.acquisition=acquisition
        self.nu=float(nu);self.thickness=float(thickness)
        self.boundary_multiplier=float(boundary_multiplier)
        nodes=acquisition.nodes;triangles=acquisition.triangles
        self.element_count=len(triangles);self.dof_count=2*len(nodes)
        points=nodes[triangles];x=points[:,:,0];y=points[:,:,1]
        determinant=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0])
        self.physical_strain_sign=-np.sign(determinant)
        self.area=.5*np.abs((x[:,0]-x[:,1])*(y[:,0]-y[:,2])-
                             (x[:,2]-x[:,0])*(y[:,1]-y[:,0]))
        if np.any(self.area<=0):raise ValueError('Degenerate triangular element')
        # Same source engineering-strain convention and arithmetic orientation.
        b=np.column_stack([y[:,2]-y[:,1],y[:,0]-y[:,2],y[:,1]-y[:,0]])
        c=np.column_stack([x[:,1]-x[:,2],x[:,2]-x[:,0],x[:,0]-x[:,1]])
        B=np.zeros((self.element_count,3,6),dtype=np.float64)
        B[:,0,::2]=b;B[:,1,1::2]=c;B[:,2,::2]=c;B[:,2,1::2]=b
        B/=2;B/=self.area[:,None,None]
        self.strain_matrix=B
        coefficient=1/(1-self.nu*self.nu)
        material=np.array([[coefficient,self.nu*coefficient,0],
                           [self.nu*coefficient,coefficient,0],
                           [0,0,1/2/(1+self.nu)]],dtype=np.float64)
        left=(self.area*self.thickness)[:,None,None]*B.transpose(0,2,1)
        self.unit_stiffness=left@(material@B)
        self.element_dofs=np.stack((2*triangles,2*triangles+1),axis=-1).reshape(-1,6)
        self.rows=np.broadcast_to(self.element_dofs[:,:,None],(self.element_count,6,6)).ravel()
        self.columns=np.broadcast_to(self.element_dofs[:,None,:],(self.element_count,6,6)).ravel()
        self.force=acquisition.nodal_force.reshape(-1).copy()
        self.fixed=np.flatnonzero(acquisition.fixed_dofs.reshape(-1))

    def assemble(self, modulus: np.ndarray):
        """Assemble CSC K(E), then replace constrained diagonals (force/length)."""
        E=np.asarray(modulus)
        if E.shape!=(self.element_count,) or E.dtype.kind!='f' or not np.isfinite(E).all() or np.any(E<=0):
            raise ValueError('Modulus must be a finite positive element vector')
        values=(E[:,None,None]*self.unit_stiffness).ravel()
        matrix=coo_matrix((values,(self.rows,self.columns)),
                          shape=(self.dof_count,self.dof_count)).tocsc()
        diagonal=matrix.diagonal()
        penalty=diagonal.max()*self.boundary_multiplier
        diagonal[self.fixed]=penalty
        matrix.setdiag(diagonal)
        return matrix

    def forward(self, modulus: np.ndarray) -> np.ndarray:
        """Return interleaved nodal displacement vector, in length units."""
        result=spsolve(self.assemble(modulus),self.force)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite FEM solution')
        return result

    def jacobian(self, modulus: np.ndarray, step: float=DIFFERENCE_STEP) -> np.ndarray:
        """Current-vector central FD, preserving the source's absolute step."""
        E=np.asarray(modulus,dtype=np.float64)
        result=np.empty((self.dof_count,self.element_count),dtype=np.float64)
        for column in range(self.element_count):
            delta=np.zeros(self.element_count);delta[column]=step
            result[:,column]=(self.forward(E+delta)-self.forward(E-delta))/(2*step)
        return result

    def strain(self, displacement: np.ndarray) -> np.ndarray:
        """Return each physical symmetric strain tensor, shape(element,2,2).

        Source B uses absolute area; its common orientation sign cancels in K.
        Restore the signed affine-gradient convention for physical strain QA.
        """
        values=np.asarray(displacement).reshape(-1)[self.element_dofs]
        engineering=self.physical_strain_sign[:,None]*np.einsum('eij,ej->ei',self.strain_matrix,values)
        result=np.zeros((self.element_count,2,2),dtype=np.float64)
        result[:,0,0]=engineering[:,0];result[:,1,1]=engineering[:,1]
        result[:,0,1]=result[:,1,0]=engineering[:,2]/2
        return result
