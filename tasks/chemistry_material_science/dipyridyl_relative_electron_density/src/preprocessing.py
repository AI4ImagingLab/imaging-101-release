"""Explicit indexed diffraction geometry and seeded amplitude initialization.

Adapted from GSAS-II, copyright2010 UChicago Argonne LLC, under the retained
Argonne open-source license. Local extraction by benchmark maintainers2026.
No upstream or crystallographic library is imported. See LICENSE/NOTICE.
"""
from pathlib import Path
import random
import numpy as np

CELL=np.array([15.8489,5.5008,23.1175,90.,96.916,90.])
ROTATIONS=np.array([np.eye(3,dtype=int),np.diag([-1,1,-1])])
TRANSLATIONS=np.array([[0.,0.,0.],[0.,.5,0.]])
GRID_STEP=.25

def load_observations(path):
    """Load the4508 real signed relative intensities, uncertainty and indices."""
    with np.load(Path(path),allow_pickle=False) as z:
        expected={'hkl':(1,4508,3),'intensity':(1,4508),'sigma_intensity':(1,4508)}
        if set(z.files)!=set(expected):raise ValueError('Unexpected observation keys')
        data={}
        for k,shape in expected.items():
            a=z[k]
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid observation '+k)
            if k=='hkl' and a.dtype.kind not in 'iu':raise ValueError('Miller indices must be integral')
            if k!='hkl' and a.dtype.kind!='f':raise ValueError('Intensities must be real floating point')
            data[k]=a[0].copy()
    if np.any(data['sigma_intensity']<=0):raise ValueError('Uncertainties must be positive')
    return data

def reciprocal_metric_coefficients(cell=CELL):
    """Return[ G11,G22,G33,2G12,2G13,2G23 ], in Angstrom^-2."""
    a,b,c,alpha,beta,gamma=np.asarray(cell,dtype=float)
    cos=lambda x:np.cos(x*np.pi/180.)
    metric=np.array([[a*a,a*b*cos(gamma),a*c*cos(beta)],
                     [a*b*cos(gamma),b*b,b*c*cos(alpha)],
                     [a*c*cos(beta),b*c*cos(alpha),c*c]])
    G=np.linalg.inv(metric)
    return np.array([G[0,0],G[1,1],G[2,2],2*G[0,1],2*G[0,2],2*G[1,2]])

def dspacings(hkl,cell=CELL):
    """All supplied Miller-index d-spacings in Angstrom; nonzero indices required."""
    h,k,l=np.asarray(hkl).T;A=reciprocal_metric_coefficients(cell)
    invd2=A[0]*h*h+A[1]*k*k+A[2]*l*l+A[3]*h*k+A[4]*h*l+A[5]*k*l
    return 1/np.sqrt(invd2)

def half_grid(dmin=.5,offset_search=False):
    """Preserve source nearest-integer dimensions and four-index padding."""
    maximum=np.round(CELL[:3]/dmin).astype(int)
    if offset_search:return 2*maximum
    return (np.ceil((maximum+1)/4.)*4).astype(int)

def equivalent_reflections(hkl):
    """P21 specialization of explicit GENHKL integer orbit/phase bookkeeping.

    The returned multiplicity follows the actual source count before adding
    Friedel partners. Negative phase offsets retain Fortran MOD sign.
    """
    h=np.asarray(hkl,dtype=int);indices=[];phase12=[];multiplicity=0;absent=0
    for R,T in zip(ROTATIONS,TRANSLATIONS):
        u=h@R;p=int(h@(12*T).astype(int));new=True;new_pair=True
        for old,oldp in zip(indices,phase12):
            if np.array_equal(u,old):
                new=False;new_pair=False
                if (p-oldp+960)%12:absent=1
                break
            if np.array_equal(u,-old):new_pair=False
        multiplicity+=int(new_pair)
        if new:indices.append(u);phase12.append(p)
    return absent,multiplicity,np.asarray(indices),np.fmod(np.asarray(phase12)/12.,1.)

def initialize_structure_factors(data,seed=20261022):
    """Expand observed amplitudes on P21 rotational orbits and Friedel partners.

    Negative measured intensities contribute zero amplitude, preserving source
    handling, while remaining publicly stored and included in residual scoring.
    No calculated structure factor, atomic coordinate or phase is read.
    """
    half=half_grid();factors=np.zeros(tuple(2*half),dtype=np.complex128)
    rng=random.Random(seed)
    for h,I,d in zip(data['hkl'],data['intensity'],dspacings(data['hkl'])):
        if not .5<d<100.:continue
        amplitude=np.sqrt(I) if I>0 else 0.
        phase=rng.uniform(0.,360.)
        indices=np.inner(h,np.array([R.T for R in ROTATIONS]));shifts=np.inner(h,TRANSLATIONS)
        for u,t in zip(indices,shifts):
            angle=phase+360*t
            value=complex(np.cos(angle*np.pi/180.),np.sin(angle*np.pi/180.))
            factors[tuple(u+half)]=amplitude*value
            factors[tuple(-u+half)]=amplitude*value.conjugate()
    return factors
