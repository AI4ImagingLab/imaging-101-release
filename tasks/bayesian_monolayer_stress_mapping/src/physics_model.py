"""Explicit staggered force balance and Bayesian prior; GPL-3.0 BISM extraction."""
import numpy as np
from scipy import sparse
def operators(rows=50,columns=50,spacing=2.,alpha_shear=1000.,alpha_boundary=1000.):
    """Return divergence A, precision B, shear-sum D, boundary selector E (CSR)."""
    n=rows*columns;nx=rows*(columns+1);ny=(rows+1)*columns
    rx=np.repeat(np.arange(n),2);ix=np.arange(n).reshape(rows,columns)
    left=(np.arange(rows)[:,None]*(columns+1)+np.arange(columns)[None,:]).ravel()
    ax=sparse.coo_matrix((np.tile([-1.,1.],n),(rx,np.column_stack([left,left+1]).ravel())),shape=(n,nx)).tocsr()
    lower=np.arange(n)
    ay=sparse.coo_matrix((np.tile([-1.,1.],n),(rx,np.column_stack([lower,lower+columns]).ravel())),shape=(n,ny)).tocsr()
    zx=sparse.csr_matrix((n,nx));zy=sparse.csr_matrix((n,ny))
    a=sparse.bmat([[ax,zy,ay,zx],[zx,ay,zy,ax]],format="csr")/spacing
    sumx=abs(ax);sumy=abs(ay)
    d=sparse.hstack([zx,zy,sumy,-sumx],format="csr")
    edge_x=np.concatenate([np.arange(rows)*(columns+1),np.arange(rows)*(columns+1)+columns])
    edge_y=np.concatenate([np.arange(columns),rows*columns+np.arange(columns)])
    selected=np.concatenate([edge_x,nx+edge_y,nx+ny+edge_y,nx+2*ny+edge_x])
    e=sparse.coo_matrix((np.ones(len(selected)),(np.arange(len(selected)),selected)),shape=(len(selected),2*nx+2*ny)).tocsr()
    b=sparse.eye(2*nx+2*ny,format="csr")+alpha_shear**2*(d.T@d)+alpha_boundary**2*(e.T@e)
    return a,b.tocsr(),d,e
def unpack_faces(vector,rows=50,columns=50):
    """Return x-faces [xx,yx] (2,R,C+1), y-faces [yy,xy] (2,R+1,C)."""
    nx=rows*(columns+1);ny=(rows+1)*columns
    xx=vector[:nx].reshape(rows,columns+1);yy=vector[nx:nx+ny].reshape(rows+1,columns)
    xy=vector[nx+ny:nx+2*ny].reshape(rows+1,columns);yx=vector[nx+2*ny:].reshape(rows,columns+1)
    return np.stack([xx,yx]),np.stack([yy,xy])
def center_stress(vector,rows=50,columns=50):
    """Interpolate physical [xx,yy,xy] to cell centers, kPa um."""
    x,y=unpack_faces(vector,rows,columns)
    return np.stack([(x[0,:,:-1]+x[0,:,1:])/2,(y[0,:-1]+y[0,1:])/2,(x[1,:,:-1]+x[1,:,1:]+y[1,:-1]+y[1,1:])/4])
def interpolation(rows=50,columns=50):
    """Return explicit sparse L mapping four face blocks to three central blocks."""
    nx=rows*(columns+1);ny=(rows+1)*columns;n=rows*columns
    left=(np.arange(rows)[:,None]*(columns+1)+np.arange(columns)[None,:]).ravel();bottom=np.arange(n)
    rr=[];cc=[];vv=[]
    for comp,indices,weights in [(0,[left,left+1],[.5,.5]),(1,[nx+bottom,nx+bottom+columns],[.5,.5]),(2,[nx+ny+bottom,nx+ny+bottom+columns,nx+2*ny+left,nx+2*ny+left+1],[.25]*4)]:
        for ids,weight in zip(indices,weights):
            rr.extend((comp*n+np.arange(n)).tolist());cc.extend(ids.tolist());vv.extend([weight]*n)
    return sparse.coo_matrix((vv,(rr,cc)),shape=(3*n,2*nx+2*ny)).tocsr()
def force_balance(vector,spacing=2.,rows=50,columns=50):
    """Independent finite-volume flux differences, shape (2,R,C), kPa."""
    x,y=unpack_faces(vector,rows,columns)
    return np.stack([(np.diff(x[0],axis=1)+np.diff(y[1],axis=0))/spacing,(np.diff(x[1],axis=1)+np.diff(y[0],axis=0))/spacing])
