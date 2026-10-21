"""Wait recursion/Jacobian adapted from Dong Hao (2011), AGPL-3.0.

Restores the missing phase chain factor documented in test/validation.md.
"""
import numpy as np

MU0=4*np.pi*1e-7
LEGACY_RHO_COEFFICIENT=7.8957e-6

def response(frequency, log_resistivity, layer_top):
    """Return log10 apparent resistivity [ohm m] and phase [rad], each (F,).

    Inputs: frequency (F,) Hz, log_resistivity (L,) log10(ohm m),
    layer_top (L,) m, increasing downward, ending in a half-space.
    """
    f=np.asarray(frequency,dtype=float); m=np.asarray(log_resistivity,dtype=float)
    z=np.asarray(layer_top,dtype=float)
    q=np.sqrt(1j*2*np.pi*f[:,None]*MU0/10**m[None,:])
    c=1/q[:,-1]
    for j in range(m.size-2,-1,-1):
        u=-2*q[:,j]*(z[j+1]-z[j]); e=np.exp(u)
        t=-np.expm1(u)/(1+e)
        c=(q[:,j]*c+t)/(q[:,j]*(1+q[:,j]*c*t))
    return np.log10(np.abs(c)**2*2*np.pi*f*MU0),np.arctan2(c.real,-c.imag)

def jacobian(frequency, log_resistivity, layer_top):
    """Return analytic (d log10(rhoa)/dm, d phase/dm, C), each (F,L).

    m=log10(resistivity); C is the complex transfer length [m] per layer.
    """
    f=np.asarray(frequency); r=10**np.asarray(log_resistivity);z=np.asarray(layer_top)
    sigma=1/r;nf=f.size;nl=r.size
    C=np.zeros((nf,nl),complex);derivative=np.ones((nf,nl),complex)
    for i,fi in enumerate(f):
        iom=1j*2*np.pi*fi*MU0
        q=np.sqrt(iom*sigma[-1]);C[i,-1]=1/q;derivative[i,-1]=sigma[-1]/(2*q)
        for j in range(nl-2,-1,-1):
            q=np.sqrt(iom*sigma[j]);h=z[j+1]-z[j]
            qp=C[i,j+1]*q+1;qm=C[i,j+1]*q-1;e=np.exp(-2*q*h)
            u=e*qm/qp;minus=1-u;plus=1+u
            derivative[i,j]=(plus/minus/q/2-2*e*(C[i,j+1]/qp**2-h*qm/qp)/minus**2)*sigma[j]
            derivative[i,j+1:]*=4*e/(qp*minus)**2
            C[i,j]=plus/minus/q
    c=C[:,0,None];norm=np.abs(c)**2
    j1=2*r[None,:]*(c.real*derivative.real+c.imag*derivative.imag)/norm
    j2=np.log(10)*r[None,:]*(c.real*derivative.imag-c.imag*derivative.real)/norm
    return j1,j2,C

def weighted_rms(predicted,observed,std):
    """Dimensionless RMS of standardized residuals."""
    return float(np.sqrt(np.mean(((np.asarray(predicted)-observed)/std)**2)))
