"""Explicit Rayleigh-Ritz free-cylinder elastodynamics from RUScal.

Adapted equations: Torres, Flores-Betancourt and Hermann, JASA151 (2022).
Original RUScal is BSD-3-Clause; see task LICENSE. No upstream imports.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import cholesky,solve_triangular,eigvalsh

DIMENSIONS_CM=np.array([1.900094,1.900094,1.338456])
MASS_G=29.7024
DENSITY_SCALED=1000*MASS_G/(DIMENSIONS_CM[0]*DIMENSIONS_CM[1]*DIMENSIONS_CM[2])*4/np.pi

def moment_table(order,dimensions=DIMENSIONS_CM):
    """Integrals of x^p y^q z^r over the cylinder, cm^(p+q+r+3)."""
    n=2*order+1
    p,q,r=np.indices((n,n,n))
    # Double factorial table, with (-1)!! and 0!! equal to one.
    df=np.ones(2*n+3)
    for k in range(4,len(df)):df[k]=df[k-2]*(k-2)
    a=np.asarray(dimensions)/2
    result=4*np.pi*np.exp((p+1)*np.log(a[0])+(q+1)*np.log(a[1])+(r+1)*np.log(a[2]))*df[p+1]*df[q+1]/((r+1)*df[p+q+4])
    result[(p%2!=0)|(q%2!=0)|(r%2!=0)]=0
    return result

def basis_blocks(order):
    """Preserve original l,m,n,component ordering and eight mirror blocks."""
    blocks=[[] for _ in range(8)]
    kx=np.arange(1,9);ky=8+((kx+1)%2)*2-kx;kz=13-8*(kx>4)-ky
    maps=[kx,ky,kz]
    for l in range(order+1):
        for m in range(order+1-l):
            for n in range(order+1-l-m):
                pk=4*(l%2)+2*(m%2)+n%2
                for component in range(3):
                    blocks[maps[component][pk]-1].append((component,l,m,n))
    return [np.asarray(b,dtype=np.int64) for b in blocks]

def _moments(table,exponents):
    valid=np.all(exponents>=0,axis=-1)
    x=np.maximum(exponents,0)
    return table[x[...,0],x[...,1],x[...,2]]*valid

def assemble_block(basis,table):
    """Return mass matrix and independent c11/c44 stiffness coefficients.

    Coefficients retain RUScal's scaled density and elastic units:
    rho=1000*(g/cm^3), tensor=10*(modulus in Mbar).
    """
    components=basis[:,0];powers=basis[:,1:]
    exponents=powers[:,None,:]+powers[None,:,:]
    same=components[:,None]==components[None,:]
    mass=DENSITY_SCALED*_moments(table,exponents)*same
    shape=mass.shape
    a=np.zeros(shape);b=np.zeros(shape)
    for j in range(3):
        for k in range(3):
            shifted=exponents.copy();shifted[...,j]-=1;shifted[...,k]-=1
            integral=_moments(table,shifted)*powers[:,j,None]*powers[None,:,k]
            normal=(components[:,None]==j)&(components[None,:]==k)
            crossed=(components[:,None]==k)&(components[None,:]==j)
            a+=normal*integral
            b+=((same if j==k else False)+crossed.astype(int)-2*normal.astype(int))*integral
    return mass,10*a,10*b

@lru_cache(maxsize=3)
def operators(order=14):
    """Build and whiten all eight blocks; cache geometry only, never answers."""
    table=moment_table(order);out=[]
    for basis in basis_blocks(order):
        mass,a,b=assemble_block(basis,table)
        lower=cholesky(mass,lower=True)
        whitened=[]
        for matrix in [a,b]:
            temp=solve_triangular(lower,matrix,lower=True)
            temp=solve_triangular(lower,temp.T,lower=True).T
            whitened.append((temp+temp.T)/2)
        out.append(tuple(whitened))
    return out

def resonances(moduli_mbar,order=14,count=76):
    """Return sorted nonrigid frequencies in MHz after six rigid modes."""
    c11,c44=np.asarray(moduli_mbar)
    values=np.concatenate([eigvalsh(c11*a+c44*b,check_finite=False) for a,b in operators(order)])
    values=np.sort(values)[6:6+count]
    return np.sqrt(np.maximum(values,0))/(2*np.pi*.1)
