"""Explicit confined-aquifer harmonic finite-volume acquisition operator.

Extraction of the author MODFLOW survey's equations; no MODFLOW/FloPy dependency.
Boundary values are imposed on boundary cell centers, not exterior cell faces.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .preprocessing import NROW,NCOL,WELL_RATE,RECHARGE

class HydraulicModel:
    def __init__(self,wells):
        self.wells=np.asarray(wells,dtype=np.int64)
        self.indices=self.wells[:,0]*NCOL+self.wells[:,1]
        self.active=np.arange(NROW*NCOL).reshape(NROW,NCOL)[:,1:-1].ravel()
        self.lookup=np.full(NROW*NCOL,-1,dtype=int);self.lookup[self.active]=np.arange(len(self.active))
        self.selection=np.flatnonzero(~np.eye(len(wells),dtype=bool).ravel())
    def assemble(self,log_conductivity,serialize=True):
        s=np.asarray(log_conductivity,dtype=np.float64).reshape(NROW,NCOL)
        if not np.isfinite(s).all():raise ValueError('Nonfinite log conductivity')
        conductivity=np.exp(s)
        if not np.isfinite(conductivity).all() or np.any(conductivity<=0):raise ValueError('Invalid conductivity')
        # Original FloPy LPF stores real32 and writes six decimal exponent digits.
        if serialize:conductivity=np.char.mod('%.6E',conductivity.astype(np.float32)).astype(np.float32).astype(np.float64)
        diagonal=np.zeros(len(self.active));rhs=np.full((len(self.active),len(self.wells)),RECHARGE*100.)
        rows=[];cols=[];values=[]
        for dy,dx in [(0,-1),(0,1),(-1,0),(1,0)]:
            iy=self.active//NCOL;ix=self.active%NCOL;jy=iy+dy;jx=ix+dx;valid=(jy>=0)&(jy<NROW)
            origin=np.flatnonzero(valid);neighbor=jy[valid]*NCOL+jx[valid]
            k1=conductivity.ravel()[self.active[valid]];k2=conductivity.ravel()[neighbor]
            conductance=2*k1*k2/(k1+k2);diagonal[origin]+=conductance
            target=self.lookup[neighbor];inside=target>=0
            rows.append(origin[inside]);cols.append(target[inside]);values.append(-conductance[inside])
            if np.any(~inside):rhs[origin[~inside]]+=conductance[~inside,None]*np.where(jx[valid][~inside,None]==0,120.,110.)
        rhs[self.lookup[self.indices],np.arange(len(self.wells))]+=WELL_RATE
        rows.append(np.arange(len(self.active)));cols.append(np.arange(len(self.active)));values.append(diagonal)
        A=sparse.csc_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(self.active),len(self.active)))
        return A,rhs
    def forward(self,log_conductivity,full=False,head_precision='single'):
        matrix,rhs=self.assemble(log_conductivity);solution=splu(matrix).solve(rhs)
        heads=np.empty((NROW*NCOL,len(self.wells)));heads[self.active]=solution
        boundary=np.arange(NROW*NCOL).reshape(NROW,NCOL)
        heads[boundary[:,0]]=120.;heads[boundary[:,-1]]=110.
        if head_precision=='single':heads=heads.astype(np.float32)
        elif head_precision!='double':raise ValueError('head_precision must be single or double')
        observed=heads[self.indices].T.ravel()[self.selection]
        return (observed,heads.T.reshape(len(self.wells),NROW,NCOL)) if full else observed

def covariance_operator(vector,length_scale=200.):
    """Stationary exponential covariance via explicit reflected circulant embedding."""
    x=np.asarray(vector,dtype=np.float64).reshape(100,75,1,order='F')
    distance=np.sqrt((np.arange(100)[:,None]*10/length_scale)**2+(np.arange(75)[None,:]*10/length_scale)**2)
    kernel=np.exp(-distance)[:,:,None]
    kernel=np.concatenate([kernel,kernel[:,-2:0:-1,:]],axis=1)
    kernel=np.concatenate([kernel,kernel[-2:0:-1,:,:]],axis=0)
    result=np.fft.ifftn(np.fft.fftn(kernel)*np.fft.fftn(x,kernel.shape,axes=(0,1,2))).real[:100,:75,:1]
    return result.reshape(-1,order='F')
