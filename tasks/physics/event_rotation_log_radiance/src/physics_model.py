"""Explicit event warping, bilinear finite differences and finite-grid derivatives.
Adapted from MIT2023 TU Berlin RIP EVILIP, revision recorded in validation.
"""
import numpy as np
from scipy import sparse
def accumulate_events(events,displacement):
    h,w=displacement.shape[1:];t,x,y,p=events.T
    # Qualified correction: floor each spatial coordinate before flattening.
    index=y.astype(np.int64)*w+x.astype(np.int64)
    velocity=displacement.reshape(2,-1)[:,index]
    xx=x+(t[-1]-t)*velocity[0];yy=y+(t[-1]-t)*velocity[1]
    out=np.zeros(h*w,dtype=np.float64);polarity=2*p-1
    for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:
        y0=np.floor(yy)+dy;x0=np.floor(xx)+dx
        weight=np.maximum(0.,1-np.abs(yy-y0))*np.maximum(0.,1-np.abs(xx-x0))
        valid=(y0>=0)&(y0<h)&(x0>=0)&(x0<w)
        np.add.at(out,(y0[valid]*w+x0[valid]).astype(np.int64),weight[valid]*polarity[valid])
    return out.reshape(h,w)
def normalized_motion(displacement):
    dx,dy=displacement;mag=np.sqrt(dx**2+dy**2);angle=np.arctan2(dy,dx)
    safe=np.where(mag>0,mag,1.)
    mask=np.zeros_like(safe);mask[1:-1,1:-1]=1
    return np.cos(angle)*mask,np.sin(angle)*mask,safe,mask
def directional_matrix(flow_y,flow_x):
    """I minus a one-pixel bilinear sample along unit motion; zero border flow."""
    h,w=flow_y.shape;y,x=flow_y.ravel(),flow_x.ravel();ay,ax=np.abs(y),np.abs(x)
    tl=(y<0)&(x<0);tr=(y<0)&(x>=0);br=(y>=0)&(x>=0);bl=(y>=0)&(x<0)
    corner=ay*ax;vertical=ay*(1-ax);horizontal=(1-ay)*ax
    diagonal=np.array([br*corner,(bl|br)*vertical,bl*corner,(tr|br)*horizontal,
        (1-ay)*(1-ax),(tl|bl)*horizontal,tr*corner,(tl|tr)*vertical,tl*corner])
    offsets=np.array([-w-1,-w,-w+1,-1,0,1,w-1,w,w+1])
    return (sparse.eye(h*w)-sparse.dia_matrix((diagonal,offsets),shape=(h*w,h*w)).T).tocsr()
def derivative_matrices(shape):
    h,w=shape
    def along(n,second=False):
        rows=[];cols=[];vals=[]
        for i in range(n):
            if second:
                j=min(max(i,1),n-2)
                for offset,val in [(-1,1.),(0,-2.),(1,1.)]:rows.append(i);cols.append(j+offset);vals.append(val)
            elif 0<i<n-1:
                for offset,val in [(-1,-.5),(1,.5)]:rows.append(i);cols.append(i+offset);vals.append(val)
        return sparse.csr_matrix((vals,(rows,cols)),shape=(n,n))
    Dy=sparse.kron(along(h),sparse.eye(w),format='csr');Dx=sparse.kron(sparse.eye(h),along(w),format='csr')
    lap=sparse.kron(along(h,True),sparse.eye(w),format='csr')+sparse.kron(sparse.eye(h),along(w,True),format='csr')
    return Dy,Dx,lap
