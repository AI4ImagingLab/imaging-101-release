"""GPL-3.0 extraction of Hendriks' explicit joint neutron/Airy mathematics.
Units: coordinates m; d0 angstrom; strains dimensionless; wavenumbers m^-1.
"""
import random
import numpy as np

def circular_modes(count):
    x,y=np.meshgrid(np.arange(1,count+1),np.arange(1,count+1));mask=np.hypot(x,y)/count<1
    return x.ravel(order='F')[mask.ravel(order='F')],y.ravel(order='F')[mask.ravel(order='F')]

def place_bases(direction,theta=(6.6359,.0223,.0044,182.4,.007,.2918),counts=(15,10),seed=1203):
    """Geometry-only spectral bases; Python MT rand matches the oracle rand stream.

    Near-zero analytical integration denominators use the author's geometric
    jitter guard. This random stream does not access or generate the target.
    """
    fd,lx,ly,fc,cx,cy=theta;m1,m2=counts;dx=4.5/lx/m1;dy=4.5/ly/m1;ex=3.5/cx/m2;ey=3.5/cy/m2
    a,b=circular_modes(m1);ldx=a*dx;ldy=b*dy;a,b=circular_modes(m2);lcx=a*ex;lcy=b*ey
    rng=random.Random(seed);shift=1e-6;n1=direction[:,0,None];n2=direction[:,1,None]
    def flags():
        tol=np.finfo(float).eps**(2/3)*np.max(lcx)
        # The source's two single-basis tests use max(lambda_d0x).
        td=np.finfo(float).eps**(2/3)*np.max(ldx)
        flag=(np.abs(n1*ldx-n2*ldy)<td)|(np.abs(n1*ldx+n2*ldy)<td)
        pdx=np.tile(ldx,len(lcx));pdy=np.tile(ldy,len(lcx));pcx=np.repeat(lcx,len(ldx));pcy=np.repeat(lcy,len(ldx))
        f=np.zeros(len(pdx),bool)
        for sx,sy,sz in [(-1,1,1),(-1,-1,-1),(-1,1,-1),(1,-1,-1),(1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1)]:
            f|=np.any(np.abs(n1*(pdx+sx*pcx)+n2*(sy*pdy+sz*pcy))<tol,axis=0)
        return np.any(flag,axis=0),f
    def draw(mask):return np.array([rng.random() for _ in range(int(np.sum(mask)))])
    for _ in range(20):
        i,j=flags()
        if not i.any() and not j.any():break
        ldx[i]+=shift*np.max(ldx)*(1-draw(i));ldy[i]+=shift*np.max(ldx)*(1-draw(i))
        mat=j.reshape(len(ldx),len(lcx),order='F');di=mat.any(1);ci=mat.any(0)
        ldx[di]+=shift*np.max(ldx)*(1-draw(di));ldy[di]+=shift*np.max(ldx)*(1-draw(di))
        lcx[ci]+=shift*np.max(ldx)*(1-draw(ci));lcy[ci]+=shift*np.max(ldx)*(1-draw(ci));shift*=np.sqrt(10)
    else:raise RuntimeError('Wavenumber guard failed')
    sd=fd**2*(2*np.pi)*lx*ly*np.exp(-.5*(ldx**2*lx**2+ldy**2*ly**2))
    sc=fc**2*(2*np.pi)*cx*cy*np.exp(-.5*(lcx**2*cx**2+lcy**2*cy**2))
    return {'ldx':ldx,'ldy':ldy,'lcx':lcx,'lcy':lcy,'sd':sd,'sc':sc,'Ldx':np.pi/2/dx,'Ldy':np.pi/2/dy,'Lcx':np.pi/2/ex,'Lcy':np.pi/2/ey,'rng':rng}

def density_basis(points,b):
    return np.sin((points[:,0,None]+b['Ldx'])*b['ldx'])*np.sin((points[:,1,None]+b['Ldy'])*b['ldy'])/np.sqrt(b['Ldx']*b['Ldy'])

def derivative_basis(points,b):
    x=(points[:,0,None]+b['Lcx'])*b['lcx'];y=(points[:,1,None]+b['Lcy'])*b['lcy'];norm=np.sqrt(b['Lcx']*b['Lcy']);s=np.sin(x)*np.sin(y)
    return np.stack((-b['lcx']**2*s/norm,-b['lcy']**2*s/norm,b['lcx']*b['lcy']*np.cos(x)*np.cos(y)/norm),axis=-1)

def strain_basis(points,b,nu=.3):
    d=derivative_basis(points,b);return np.stack((d[:,:,1]-nu*d[:,:,0],-(1+nu)*d[:,:,2],d[:,:,0]-nu*d[:,:,1]),axis=1).reshape(3*len(points),-1)

def traction_basis(points,normals,b):
    d=derivative_basis(points,b);n1=normals[:,0,None];n2=normals[:,1,None]
    return np.stack((n1*d[:,:,1]-n2*d[:,:,2],-n1*d[:,:,2]+n2*d[:,:,0]),axis=1).reshape(2*len(points),-1)

def integral_density(entry,exit,direction,b):
    ax=(entry[:,0,None]+b['Ldx'])*b['ldx'];ay=(entry[:,1,None]+b['Ldy'])*b['ldy'];bx=(exit[:,0,None]+b['Ldx'])*b['ldx'];by=(exit[:,1,None]+b['Ldy'])*b['ldy']
    q1=direction[:,0,None]*b['ldx']-direction[:,1,None]*b['ldy'];q2=direction[:,0,None]*b['ldx']+direction[:,1,None]*b['ldy']
    return ((np.sin(bx-by)-np.sin(ax-ay))/(2*q1)-(np.sin(bx+by)-np.sin(ax+ay))/(2*q2))/np.sqrt(b['Ldx']*b['Ldy'])

def integral_products(entry,exit,direction,b):
    """Analytic integrals of density sine basis × three Airy second derivatives.

    Return (rays,density_modes*stress_modes,3), density mode varying fastest.
    """
    dx=np.tile(b['ldx'],len(b['lcx']));dy=np.tile(b['ldy'],len(b['lcx']));cx=np.repeat(b['lcx'],len(b['ldx']));cy=np.repeat(b['lcy'],len(b['ldx']))
    def phase(p):return [(p[:,0,None]+b['Ldx'])*dx,(p[:,1,None]+b['Ldy'])*dy,(p[:,0,None]+b['Lcx'])*cx,(p[:,1,None]+b['Lcy'])*cy]
    a=phase(entry);z=phase(exit);n1=direction[:,0,None];n2=direction[:,1,None];g=[]
    for sx,sy,sz in [(-1,1,1),(-1,-1,-1),(-1,1,-1),(1,-1,-1),(1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1)]:
        start=a[0]+sx*a[2]+sy*a[1]+sz*a[3];end=z[0]+sx*z[2]+sy*z[1]+sz*z[3]
        g.append((np.sin(end)-np.sin(start))/(8*(n1*dx+sx*n1*cx+sy*n2*dy+sz*n2*cy)))
    g1,g2,g3,g4,g5,g6,g7,g8=g
    o1=-g1-g2+g3+g4-g5+g6-g7+g8;o2=-g1+g2-g3+g4-g5+g6+g7-g8;norm=np.sqrt(b['Lcx']*b['Lcy']*b['Ldx']*b['Ldy'])
    return np.stack((-cx**2/norm*o1,-cy**2/norm*o1,cx*cy/norm*o2),axis=-1)

def acquisition_operators(data,b,chunk=128,nu=.3):
    """Build explicit linear-density and bilinear strain-density line averages."""
    e=data['entry'];x=data['exit'];n=data['direction'];length=data['path_length'];m1=len(b['ldx']);m2=len(b['lcx']);bilinear=np.empty((len(e),m1,m2))
    for k in range(0,len(e),chunk):
        sl=slice(k,k+chunk);ip=integral_products(e[sl],x[sl],n[sl],b);v=n[sl];l=length[sl]
        t=(v[:,0,None]**2*(ip[:,:,1]-nu*ip[:,:,0])-2*(1+nu)*(v[:,0]*v[:,1])[:,None]*ip[:,:,2]+v[:,1,None]**2*(ip[:,:,0]-nu*ip[:,:,1]))/l[:,None]
        bilinear[sl]=t.reshape(-1,m1,m2,order='F')
    linear=integral_density(e,x,n,b)/length[:,None]
    return linear,bilinear

def prediction(weights,linear,bilinear):
    m=linear.shape[1];d=weights[:m];c=weights[m:]
    return linear@d+np.einsum('njk,j,k->n',bilinear,d,c,optimize=True)

def jacobian(weights,linear,bilinear):
    m=linear.shape[1];return np.c_[linear+np.einsum('njk,k->nj',bilinear,weights[m:],optimize=True),np.einsum('njk,j->nk',bilinear,weights[:m],optimize=True)]
