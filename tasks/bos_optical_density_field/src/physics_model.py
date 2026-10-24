"""Explicit thin-slab BOS calibration and edge-discretized integration operators.

Selected Rajendran et al. WLS methodology; minimum uncertainty calibration
correction is documented in validation. Independent local mathematical extraction.
"""
import numpy as np
from scipy import sparse

RHO_AMBIENT=1.225
BOUNDARY_SIGMA=1e-10
PIXEL_PITCH=1e-5
GLADSTONE_DALE=.225e-3
REFRACTIVE_AMBIENT=1+GLADSTONE_DALE*RHO_AMBIENT
DEPTH=.01
DISTANCE=.25
MAGNIFICATION=.105/(1-.105)
GRADIENT_MAGNIFICATION=MAGNIFICATION/.75
MAGNIFICATION_SIGMA=.1
DISTANCE_SIGMA=.001

def calibrate(Xpix,Ypix,U,V,sU,sV):
    """Return x,y[m],gx,gy,sgx,sgy[kg/m4] on the complete native grid."""
    arrays=[np.asarray(a,dtype=np.float64) for a in (Xpix,Ypix,U,V,sU,sV)]
    if any(a.shape!=arrays[0].shape or a.ndim!=2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Calibration requires equal finite two-dimensional arrays')
    Xpix,Ypix,U,V,sU,sV=arrays
    if np.any(sU<=0) or np.any(sV<=0):raise ValueError('Displacement uncertainty must be positive')
    scale=REFRACTIVE_AMBIENT*PIXEL_PITCH/(MAGNIFICATION*GLADSTONE_DALE*DISTANCE*DEPTH)
    def uncertainty(a,s):
        return np.sqrt((scale*s)**2+(scale*a*MAGNIFICATION_SIGMA/MAGNIFICATION)**2+(scale*a*DISTANCE_SIGMA/DISTANCE)**2)
    return ((Xpix-256)*PIXEL_PITCH/GRADIENT_MAGNIFICATION,
            (Ypix-256)*PIXEL_PITCH/GRADIENT_MAGNIFICATION,
            U*scale,V*scale,uncertainty(U,sU),uncertainty(V,sV))

def perimeter(shape):
    """Return row-major full perimeter selector, with corners counted once."""
    mask=np.ones(shape,dtype=bool);mask[1:-1,1:-1]=False
    return mask

def grid_spacing(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if x.ndim!=2 or x.shape!=y.shape or min(x.shape)<2:raise ValueError('Grid must be rectangular2D with each size>=2')
    dx=float(x[1,1]-x[0,0]);dy=float(y[1,1]-y[0,0])
    if dx<=0 or dy<=0 or not np.allclose(x,x[0:1,:],rtol=0,atol=1e-13) or not np.allclose(y,y[:,0:1],rtol=0,atol=1e-13):raise ValueError('Expected increasing Cartesian coordinates')
    if not np.allclose(np.diff(x,axis=1),dx) or not np.allclose(np.diff(y,axis=0),dy):raise ValueError('Expected uniform grid')
    return dx,dy

def integration_operators(x,y):
    """Return D edge derivatives, B endpoint averaging, C boundary scaling, CSR."""
    dx,dy=grid_spacing(x,y);ny,nx=x.shape;n=ny*nx
    nodes=np.arange(n).reshape(ny,nx)
    west=nodes[:,:-1].ravel();east=nodes[:,1:].ravel()
    south=nodes[:-1,:].ravel();north=nodes[1:,:].ravel()
    boundary=np.flatnonzero(perimeter(x.shape));nedge=len(west)+len(south);nb=len(boundary)
    edge_rows=np.repeat(np.arange(nedge),2)
    edge_cols=np.stack((np.r_[west,south],np.r_[east,north]),axis=1).ravel()
    inv=np.r_[np.full(len(west),1/dx),np.full(len(south),1/dy)]
    vals=np.stack((-inv,inv),axis=1).ravel()
    D=sparse.coo_matrix((np.r_[vals,np.full(nb,1/dx)],(np.r_[edge_rows,np.arange(nedge,nedge+nb)],np.r_[edge_cols,boundary])),shape=(nedge+nb,n)).tocsr()
    gradient_cols=edge_cols.copy();gradient_cols[2*len(west):]+=n
    B=sparse.coo_matrix((np.full(2*nedge,.5),(edge_rows,gradient_cols)),shape=(nedge+nb,2*n)).tocsr()
    C=sparse.coo_matrix((np.full(nb,1/dx),(np.arange(nedge,nedge+nb),np.arange(nb))),shape=(nedge+nb,nb)).tocsr()
    return D,B,C

def poisson_operators(x,y):
    """Author centered divergence/Laplacian with the selected full Dirichlet border."""
    dx,dy=grid_spacing(x,y);ny,nx=x.shape;n=ny*nx;bc=perimeter(x.shape)
    rr=[];cc=[];vv=[];xr=[];xc=[];xv=[];yr=[];yc=[];yv=[]
    for j in range(ny):
        for i in range(nx):
            k=j*nx+i
            if bc[j,i]:rr.append(k);cc.append(k);vv.append(1/dx**2);continue
            for stride,h,rows,cols,vals in [(1,dx,xr,xc,xv),(nx,dy,yr,yc,yv)]:
                rr.extend([k,k,k]);cc.extend([k-stride,k,k+stride]);vv.extend([1/h**2,-2/h**2,1/h**2])
                rows.extend([k,k]);cols.extend([k-stride,k+stride]);vals.extend([-.5/h,.5/h])
    A=sparse.coo_matrix((vv,(rr,cc)),shape=(n,n)).tocsr()
    Hx=sparse.coo_matrix((xv,(xr,xc)),shape=(n,n)).tocsr();Hy=sparse.coo_matrix((yv,(yr,yc)),shape=(n,n)).tocsr()
    C=sparse.diags(bc.ravel().astype(float)/dx**2,format='csr')
    return A,Hx,Hy,C
