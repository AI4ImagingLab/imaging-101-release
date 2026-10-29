"""Explicit Hermite POVMs, positive ML update and canonical Wigner field.

Core recurrences adapted from Xanadu GKP author analysis_functions.py,
revision5a49ea80a3d3776042474a154e335246291ece8c, Apache-2.0.
Independent original capture and equation/coordinate provenance are in validation.
"""
import numpy as np
from scipy.linalg import sqrtm
from scipy.special import eval_genlaguerre,gammaln

HBAR=1.0
PHOTON_CUTOFF=30
Q_GRID=np.linspace(-7,7,201)
P_GRID=np.linspace(-7,7,201)

def hermite_amplitudes(q,theta,cutoff=30,hbar=1):
    """Return <n|q,theta> array(cutoff+1,N); q variance of vacuum=hbar/2."""
    q=np.asarray(q,dtype=np.float64);theta=np.asarray(theta,dtype=np.float64)
    if q.ndim!=1 or theta.shape!=q.shape or cutoff<1 or hbar<=0:
        raise ValueError('Equal one-dimensional samples, cutoff>=1 and positive hbar required')
    a=np.zeros((cutoff+1,len(q)),dtype=complex)
    a[0]=(1/(np.pi*hbar)**(1/4))*np.exp(-(q**2)/(2*hbar))
    a[1]=(np.sqrt(2/hbar)/((hbar*np.pi)**(1/4)))*q*np.exp(-(q**2)/(2*hbar))*np.exp(1j*theta)
    for i in range(2,cutoff+1):
        a[i]=np.sqrt(2/hbar)/np.sqrt(i)*q*a[i-1]*np.exp(1j*theta)-(i-1)/np.sqrt(i*(i-1))*a[i-2]*np.exp(2j*theta)
    return a

def born_probabilities(density,amplitudes):
    """Continuous probability density at each measured conditional quadrature."""
    return np.sum(amplitudes*(density.T@amplitudes.conj()),axis=0)

def log_likelihood(density,amplitudes):
    return float(np.real(np.sum(np.log(born_probabilities(density,amplitudes)))))

def density_update(density,amplitudes):
    """One native R rho R update, returning new density, R and old Born density."""
    p=born_probabilities(density,amplitudes)
    R=(amplitudes/p)@amplitudes.conj().T
    R=(R+R.conj().T)/2
    result=R@density@R
    result=(result+result.conj().T)/2
    result=result/result.trace()
    return result,R,p

def step_fidelity(density,previous):
    """Retain author's normalized-state matrix-square-root diagnostic convention."""
    root=sqrtm(density)
    root=root/np.trace(root@root)
    return np.trace(sqrtm(root@previous@root))**2

def wigner_function(density,q,p,hbar=1):
    """Native explicit stable polynomial recurrence, output rows p, columns q."""
    N=len(density)
    X,P=np.meshgrid(np.asarray(q)/np.sqrt(hbar),np.asarray(p)/np.sqrt(hbar))
    alpha=np.exp(-(X**2)-P**2)/np.pi
    beta=X-1j*P;gamma=2*X**2+2*P**2
    field=1j*np.zeros(alpha.shape)
    v,w=alpha,alpha*(gamma-1)
    field+=density[0,0]*v+density[1,1]*w
    V,W=v,w
    for n in range(2,N):
        v,w=w,-np.sqrt((n-1)*(n-1)/(n*n))*v-(2*n-1-gamma)/np.sqrt(n*n)*w
        field+=density[n,n]*w
    for d in range(1,N-1):
        v,w=beta*np.sqrt(2/d)*V,-beta*np.sqrt(2/(d*(d+1)))*V+beta*np.sqrt(2/(d+1))*W
        field+=2*np.real(density[d,0]*v)+2*np.real(density[d+1,1]*w)
        V,W=v,w
        for n in range(2,N-d):
            v,w=w,-np.sqrt((n-1)*(d+n-1)/(n*(d+n)))*v-(2*n+d-1-gamma)/np.sqrt(n*(d+n))*w
            field+=2*np.real(density[d+n,n]*w)
    v=beta*np.sqrt(2/(N-1))*V
    field+=2*np.real(density[N-1,0]*v)
    return np.real_if_close(field)/hbar

def quadrature_probability(density,q,theta):
    """Native finite-grid normalized probability mass, not continuous PDF."""
    q=np.asarray(q);n_photon=len(density)-1
    H=np.ones((n_photon+1,len(q)));H[1]=2*q
    factorial=np.ones(n_photon+1)
    for n in range(2,n_photon+1):
        H[n]=2*q*H[n-1]-2*(n-1)*H[n-2]
        factorial[n]=factorial[n-1]*n
    probability=np.zeros(len(q),dtype=complex)
    for n in range(len(density)):
        for m in range(len(density)):
            probability+=density[n,m]*H[n]*H[m]*np.exp(-(q**2))*np.exp(1j*(m-n)*theta)/np.sqrt(2**(n+m)*factorial[n]*factorial[m]*np.pi)
    return np.real_if_close(probability)/np.sum(probability.real)

def displacement(shape,alpha):
    """Closed generalized-Laguerre displacement matrix; general special functions only."""
    rows,columns=shape
    if alpha==0:return np.eye(rows,columns,dtype=complex)
    result=np.empty(shape,dtype=complex);r2=abs(alpha)**2
    for n in range(rows):
        for m in range(columns):
            if n>=m:
                result[n,m]=np.exp(-r2/2+(gammaln(m+1)-gammaln(n+1))/2)*alpha**(n-m)*eval_genlaguerre(m,n-m,r2)
            else:
                result[n,m]=np.exp(-r2/2+(gammaln(n+1)-gammaln(m+1))/2)*(-alpha.conjugate())**(m-n)*eval_genlaguerre(n,m-n,r2)
    return result
