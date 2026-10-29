"""PWL DRT discretization adapted from Ciucci et al.; see task LICENSE."""
import numpy as np

def operators(frequency):
    """Return (complex design (N,N+2), derivative (N-1,N+2), tau (N,))."""
    f=np.asarray(frequency,dtype=np.float64);tau=1/f;u=np.log(tau)
    weights=np.empty_like(u);weights[0]=(u[1]-u[0])/2;weights[-1]=(u[-1]-u[-2])/2;weights[1:-1]=(u[2:]-u[:-2])/2
    w=2*np.pi*f
    kernel=weights[None,:]/(1+1j*w[:,None]*tau[None,:])
    design=np.column_stack([1j*w,np.ones(len(f)),kernel])
    derivative=np.zeros((len(f)-1,len(f)+2))
    # Upstream PWL derivative spacing uses log(1/tau), reversing both signs.
    delta=np.log(tau[:-1]/tau[1:])
    i=np.arange(len(f)-1);derivative[i,i+2]=-1/delta;derivative[i,i+3]=1/delta
    return design,derivative,tau

def forward(parameters,frequency):
    return operators(frequency)[0]@parameters

def objective(parameters,frequency,impedance,regularization=1e-3):
    a,l,_=operators(frequency);r=a@parameters-impedance
    return float(np.vdot(r,r).real+regularization*np.sum((l@parameters)**2))
