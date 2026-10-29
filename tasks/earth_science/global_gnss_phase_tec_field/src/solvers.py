"""Explicit weighted normal equations and author one-pass negative-node LCP.

MosGIM2 domain stages: MIT, Copyright2023 Artem Padokhin/Artem Vesnin.
Lemke tableau stages adapted from MIT lemkelcp0.1, Andy Lamperski.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix
from .physics_model import (design_matrix,chunk_bounds,temporal_precision,
                            grid_basis,render_maps,DEGREE,INTERVALS,SIGMA_REL)

class LemkeTableau:
    """Complementary pivot tableau [I,-M,-1,q] with strict source ratio ties."""
    def __init__(self,M,q):
        self.n=len(q);n=self.n
        self.table=np.hstack((np.eye(n),-M,-np.ones((n,1)),q[:,None]))
        self.labels=np.hstack((np.vstack((np.zeros(n,dtype=int),np.arange(n))),
                               np.vstack((np.ones(n,dtype=int),np.arange(n))),
                               np.array([[2,3],[0,0]])))
        self.wpos=list(range(n));self.zpos=list(range(n,2*n))
    def swap(self,a,b):
        for old,new in [(a,b),(b,a)]:
            kind,j=self.labels[:,old]
            if kind==0:self.wpos[j]=new%(2*self.n+2)
            elif kind==1:self.zpos[j]=new%(2*self.n+2)
        self.labels[:,[a,b]]=self.labels[:,[b,a]]
        self.table[:,[a,b]]=self.table[:,[b,a]]
    def pivot(self,row):
        kind,j=self.labels[:,row]
        partner=self.zpos[j] if kind==0 else self.wpos[j] if kind==1 else None
        if partner is not None:self.swap(row,partner)
        self.swap(row,-2)
    def eliminate(self,row):
        self.table[row]/=self.table[row,-2]
        for i in range(self.n):
            if i!=row:self.table[i]-=self.table[i,-2]*self.table[row]
    def initialize(self):
        if self.table[:,-1].min()>=0:return False
        row=int(self.table[:,-1].argmin());self.eliminate(row);self.pivot(row)
        return True
    def step(self):
        row=None;ratio=np.inf
        for i in range(self.n):
            a=self.table[i,-2]
            if a>0:
                trial=self.table[i,-1]/a
                if trial<ratio:row=i;ratio=trial
        if row is None:return False
        self.eliminate(row);self.pivot(row);return True
    def solution(self):
        z=np.zeros(self.n)
        for i in range(self.n):
            if self.labels[0,i]==1:z[self.labels[1,i]]=self.table[i,-1]
        return z

def lemke(M,q,max_iterations=1000000):
    """Solve Mz+q>=0,z>=0,z·(Mz+q)=0; raise on source failure statuses."""
    t=LemkeTableau(np.asarray(M),np.asarray(q))
    if not t.initialize():return np.zeros(len(q))
    for _ in range(max_iterations):
        success=t.step()
        if t.labels[0,-2]==2:return t.solution()
        if not success:raise RuntimeError('Lemke secondary ray')
    raise RuntimeError('Lemke iteration budget exceeded')

def assemble_normal_system(data, sigma_rel=SIGMA_REL, progress=False):
    """All rays, original168chunk summation, then 24 hourly difference priors."""
    size=(INTERVALS+1)*(DEGREE+1)**2
    N=np.zeros((size,size));b=np.zeros(size);rPr=0.;count=len(data['rhs'])
    bounds=chunk_bounds(count)
    for j,(start,end) in enumerate(bounds):
        d={key:value[start:end] for key,value in data.items()};A,w=design_matrix(d)
        AP=A.T.multiply(w[None,:]).tocsr()
        N+=(AP@A).toarray();b+=AP@d['rhs'];rPr+=np.dot(w,d['rhs']**2)
        if progress and (j%25==0 or j==len(bounds)-1):print(f'normal chunks {j+1}/{len(bounds)}',flush=True)
    N+=temporal_precision(sigma_rel=sigma_rel)
    dof=-size+count+INTERVALS*(DEGREE+1)**2
    return N,b,float(rPr),int(dof)

def negative_node_correction(coefficients,inverse_normal):
    """Author single pass: constrain only grid points initially below zero.

    This is deliberately not an all-node active-set re-solve; new tiny negative
    values elsewhere remain part of the evaluated signed output.
    """
    nc=(DEGREE+1)**2;B=grid_basis();w=(coefficients.reshape(25,nc)@B.T).ravel()
    ids=np.flatnonzero(w<0)
    if not len(ids):return coefficients.copy(),np.zeros(0),ids
    hours=ids//len(B);sites=ids%len(B)
    G=csr_matrix((B[sites].ravel(),(np.repeat(np.arange(len(ids)),nc),
            (hours[:,None]*nc+np.arange(nc)[None,:]).ravel())),shape=(len(ids),len(coefficients)))
    NGT=inverse_normal@G.T
    M=np.asarray(G@NGT);dual=lemke(M,w[ids])
    return coefficients+NGT@dual,dual,ids

def solve_system(N,b,rhs_weighted_square,dof, nonnegative=True):
    """Direct inverse as selected source, followed by optional one-pass LCP."""
    inverse=np.linalg.inv(N);unconstrained=inverse@b
    dispersion=float((rhs_weighted_square-unconstrained@b)/dof)
    if nonnegative:c,dual,ids=negative_node_correction(unconstrained,inverse)
    else:c=unconstrained.copy();dual=np.zeros(0);ids=np.zeros(0,dtype=int)
    return dict(coefficients=c,unconstrained_coefficients=unconstrained,dual=dual,
                initial_negative_indices=ids,dispersion=dispersion)

def reconstruct(data,progress=False):
    """Fresh full benchmark; returns field, coefficients and source diagnostics."""
    start=time.perf_counter();N,b,rPr,dof=assemble_normal_system(data,progress=progress)
    result=solve_system(N,b,rPr,dof)
    result['reconstruction']=render_maps(result['coefficients'])[None]
    result['runtime_seconds']=time.perf_counter()-start
    return result
