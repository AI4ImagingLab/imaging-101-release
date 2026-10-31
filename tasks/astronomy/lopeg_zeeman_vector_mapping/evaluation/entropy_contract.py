"""Independent vector-harmonic/entropy contract; no reconstruction imports."""
import numpy as np
from scipy.special import lpmv, gammaln

def harmonics(theta,phi):
    pairs=[(l,m) for l in range(1,16) for m in range(l+1)]
    Y=[];Z=[];X=[]
    q=np.cos(theta);sn=np.sin(theta)
    for l,m in pairs:
        P=lpmv(m,l,q);previous=lpmv(m,l-1,q) if m<l else np.zeros_like(q)
        derivative=-sn*(l*q*P-(l+m)*previous)/(q*q-1)
        c=np.sqrt((2*l+1)/(4*np.pi)*np.exp(gammaln(l-m+1)-gammaln(l+m+1)))
        phase=np.exp(1j*m*phi);Y.append(c*P*phase);Z.append(c*derivative*phase/(l+1));X.append(1j*m*c*P*phase/((l+1)*sn))
    return np.asarray(Y),np.asarray(Z),np.asarray(X),pairs

def field(coeff,theta,phi):
    y,z,x,_=harmonics(theta,phi);a,b,c=coeff
    return np.asarray([(a@y).real,-(b@z+c@x).real,-(b@x-c@z).real])

def coefficients(values,theta,phi):
    """Invert ring Fourier series then small latitude LS; no target coefficient use."""
    Y,Z,X,pairs=harmonics(theta,phi);rings=np.unique(theta);selected=[]
    for angle in rings:
        idx=np.flatnonzero(theta==angle)
        if len(idx)>30:selected.append(idx)
    coeff=np.zeros((3,135),dtype='complex128')
    for m in range(16):
        cols=[i for i,(_,order) in enumerate(pairs) if order==m]
        yr=[];zr=[];xr=[];observed=[]
        for idx in selected:
            phase=np.exp(-1j*m*phi[idx]);factor=1 if m==0 else 2
            observed.append(factor*np.mean(values[:,idx]*phase,axis=1))
            point=idx[0];e=np.exp(-1j*m*phi[point]);yr.append((Y[cols,point]*e).real);zr.append((Z[cols,point]*e).real);xr.append((X[cols,point]*e).imag)
        yr=np.asarray(yr);zr=np.asarray(zr);xr=np.asarray(xr);observed=np.asarray(observed)
        coeff[0,cols]=np.linalg.lstsq(yr,observed[:,0],rcond=None)[0]
        matrix=np.block([[-zr,-1j*xr],[-1j*xr,zr]])
        sol=np.linalg.lstsq(matrix,np.r_[observed[:,1],observed[:,2]],rcond=None)[0]
        coeff[1,cols]=sol[:len(cols)];coeff[2,cols]=sol[len(cols):]
    return coeff

def entropy_cost(coeff):
    degree=np.asarray([l for l in range(1,16) for m in range(l+1)])
    packed=np.concatenate([coeff.real.ravel(),coeff.imag.ravel()]);w=np.tile(degree,6);psi=np.hypot(packed,200.)
    return float(np.sum(w*(packed*np.arcsinh(packed/200.)-psi+200.)))
