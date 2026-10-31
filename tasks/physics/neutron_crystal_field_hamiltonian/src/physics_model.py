"""Explicit Stevens Hamiltonian and finite-band neutron response.

Adapted from Allen Scheie's PyCrystalField, GPL-3.0; see LICENSE/NOTICE.
Only general NumPy/SciPy numerical primitives are used.
"""
import numpy as np
from scipy.linalg import eig_banded,eigh
from scipy.linalg.blas import ddot,dgemv
from scipy.special import wofz

J=7.5
def power(a,n):
    result=a.copy()
    for _ in range(n-1):result=np.dot(result,a)
    return result

def angular_momentum():
    m=np.arange(-J,J+1);jp=np.zeros((16,16))
    for k in range(15):jp[k+1,k]=np.sqrt((J-m[k])*(J+m[k]+1))
    jm=jp.T;return .5*(jp+jm),-.5*(jp-jm),np.diag(m),jp,jm

def stevens_operators():
    _,_,z,p,m=angular_momentum();X=J*(J+1);I=np.eye(16)
    z2=power(z,2);z3=power(z,3);z4=power(z,4);z6=power(z,6);s3=power(p,3)+power(m,3)
    q=11*z3-(3*X+59)*z
    return np.array([3*z2-X*I,35*z4-(30*X-25)*z2+3*X**2*I-6*X*I,
      .25*(s3@z+z@s3),231*z6-(315*X-735)*z4+(105*X**2-525*X+294)*z2-5*X**3*I+40*X**2*I-60*X*I,
      .25*(s3@q+q@s3),.5*(power(p,6)+power(m,6))])

OPERATORS=stevens_operators()
JX,JY_IMAG,JZ,_,_=angular_momentum()

def diagonalize(coefficients):
    H=np.sum([o*b for o,b in zip(OPERATORS,coefficients)],axis=0)
    bands=np.zeros((16,16),complex);last=0
    for i in range(16):
        d=H.diagonal(i)
        if i==0:bands[i]=d
        else:bands[i][:-i]=d
        if np.count_nonzero(np.around(d,10))>0:last=i
    e,v=eig_banded(bands[:last+1],lower=True);e=e-np.min(e);v=v.T
    e[abs(e)<1e-15]=0.;v[abs(v)<1e-15]=0.
    return H,e,v

def transition(v,w):
    # SciPy BLAS preserves the original Numba BLAS reduction convention.
    ax=ddot(v,dgemv(1.,JX.T,w,trans=1))
    ay=ddot(v,dgemv(1.,JY_IMAG.T,w,trans=1))
    az=ddot(v,dgemv(1.,JZ.T,w,trans=1))
    return ax*ax+ay*ay+az*az

def voigt(x,center,alpha,gamma):
    """Source signed-width formula, validated only on the bound energy domain."""
    sigma=(.5*alpha)/np.sqrt(2*np.log(2))
    return np.real(wofz(((x-center)+1j*(.5*gamma))/sigma/np.sqrt(2)))/sigma/np.sqrt(2*np.pi)

def neutron_spectrum(energy,temperature,width,gamma,eigenvalues,eigenvectors):
    intensity=np.zeros(len(energy));beta=1/(8.61733e-2*temperature)
    Z=sum([np.exp(-beta*en) for en in eigenvalues])
    for i,ket in enumerate(eigenvectors.real):
        pn=np.exp(-beta*eigenvalues[i])/Z
        if pn>1e-3:
            for j,other in enumerate(eigenvectors.real):
                de=eigenvalues[j]-eigenvalues[i]
                intensity+=((pn*transition(ket,other)*voigt(energy,de,width[0]*de+width[1],gamma)).real).astype('float64')
    return intensity

def predict(parameters,energy,temperature,width):
    _,e,v=diagonalize(parameters[:6]);curves=[]
    for k,T in enumerate(temperature):
        curves.append(parameters[6]*neutron_spectrum(energy[k]*(1+T*parameters[14])-parameters[10],T,width,parameters[7+k],e,v)+parameters[11+k])
    return np.array(curves)

def magnetization(coefficients,temperature,field):
    H,_,_=diagonalize(coefficients);B=np.asarray(field);g=1.2;mu=.057883818012
    e,v=eigh(H+g*mu*(B[0]*JX+B[1]*JY_IMAG*1j+B[2]*JZ));e-=np.min(e);v=v.T
    expectations=np.array([[np.real(np.dot(np.conjugate(w),np.dot(A,w))) for A in [JX,JY_IMAG*1j,JZ]] for w in v])
    weights=np.exp(-e/(.086173303*temperature))
    return g*np.dot(weights,expectations)/np.sum(weights)

def gtensor(coefficients):
    """Original ground-doublet projection; row norms are physical axis g values."""
    _,e,v=diagonalize(coefficients);ground=v[np.where(np.around(e,5)==0)]
    if len(ground)!=2:raise ValueError('Non-doublet ground state')
    v1,v2=np.around(ground,10)
    def element(A,w):
        value=np.dot(v1,np.dot(A,np.conj(w)));rounded=np.around(value,10)
        return float(rounded.real) if rounded.imag==0 else value
    rows=[]
    for axis,A in enumerate([JX,JY_IMAG*1j,JZ]):
        cross=element(A,v2);diagonal=element(A,v1)
        rows.append([np.real(cross),np.imag(cross),abs(diagonal) if axis==2 else diagonal])
    return 2*np.array(rows)*1.2
