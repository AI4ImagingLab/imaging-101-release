"""Local FFT acquisition, sign-flip dynamics and original origin heuristic.

GSAS-II mathematical extraction, copyright2010 UChicago Argonne LLC; adapted
by benchmark maintainers2026 under the complete license in LICENSE/NOTICE.
"""
import numpy as np
from .preprocessing import CELL,half_grid,equivalent_reflections

def cell_basis():
    """Return lattice vectors as Cartesian columns, in Angstrom."""
    a,b,c,alpha,beta,gamma=CELL
    ca,cb,cg=np.cos(np.array([alpha,beta,gamma])*np.pi/180.)
    sg=np.sin(gamma*np.pi/180.)
    return np.array([[a,b*cg,c*cb],[0.,b*sg,c*(ca-cb*cg)/sg],
                     [0.,0.,c*np.sqrt(1-cb*cb-((ca-cb*cg)/sg)**2)]])

def structure_factors(density,hkl):
    """Evaluate F(h)=10/N sum_j rho(j) exp(+2pi i h.j/shape)."""
    rho=np.asarray(density);h=np.asarray(hkl,dtype=int)
    F=np.fft.ifftn(10*rho)
    return F[tuple((h%np.asarray(rho.shape)).T)]

def intensity_relative_error(amplitudes,intensities):
    """Relative L2 error against every signed measured intensity, no censoring."""
    I=np.asarray(intensities);den=np.linalg.norm(I)
    if den==0:raise ValueError('Undefined intensity normalization')
    return float(np.linalg.norm(abs(amplitudes)**2-I)/den)

def flip_density(density,k_factor=.1,k_max=30.):
    """Pointwise sign flips below k_factor*sigma or above k_max*sigma."""
    rho=np.asarray(density);sigma=np.std(rho)
    result=np.where(rho>=k_factor*sigma,rho,-rho)
    return np.where(rho<=k_max*sigma,result,-result)

def charge_flip_update(state,observed_amplitudes,k_factor=.1,k_max=30.):
    """One complete real/reciprocal update, retaining the source operation order."""
    rho=np.real(np.fft.fftn(np.fft.fftshift(state)))*(1.+0j)
    sigma=np.std(rho)
    flipped=np.where(np.real(rho)>=k_factor*sigma,rho,-rho)
    flipped=np.where(np.real(rho)<=k_max*sigma,flipped,-flipped)
    temporary=np.fft.ifftshift(np.fft.ifftn(flipped))
    temporary=np.where(temporary,temporary,1.)
    phase=temporary/abs(temporary)
    new=observed_amplitudes*phase
    mask=observed_amplitudes==0
    total=np.sum(np.ma.array(observed_amplitudes,mask=mask))
    current_total=np.sum(np.ma.array(abs(temporary),mask=mask))
    if total<=0 or current_total<=0:raise ValueError('Nonzero observations required')
    difference=abs(observed_amplitudes/total-abs(temporary)/current_total)
    residual=min(100.,float(np.sum(np.ma.array(difference,mask=mask)*100.)))
    return new,{'density':rho.real,'sigma':float(sigma),'flipped_density':flipped.real,
                'temporary_factors':temporary,'phase':phase,'residual_percent':residual}

def source_origin_shift(factors):
    """Preserve the source's heuristic symmetry-phase origin choice explicitly.

    Decimal string amplitude keys, complex sorting and meshgrid conventions
    are retained rather than silently replaced with a different origin solver.
    The scientific density comparison independently allows periodic translation.
    This heuristic is not a proven crystallographic origin calibration.
    """
    shape=factors.shape;half=np.asarray(shape)//2
    by_amplitude={}
    for index in np.argsort(factors.flatten()):
        u=np.unravel_index(index,shape);value=factors[u]
        if value==0.:break
        by_amplitude['%.6f'%abs(value)]=index
    names=np.flipud(np.sort(list(by_amplitude.keys())));DH=[];Dphi=[];count=0
    maximum=half_grid(3.5,offset_search=True)
    for key in names:
        u=np.array(np.unravel_index(by_amplitude[key],shape))
        if np.any(abs(u-half)-maximum>0):continue
        _,_,unique,phase=equivalent_reflections(u-half)
        unique=np.concatenate((unique,-unique))+half;phase=np.concatenate((phase,-phase))
        angle0=np.angle(factors[tuple(u)],deg=True)/360.
        for H,p in list(zip(unique,phase))[1:]:
            angle=np.angle(factors[tuple(H)],deg=True)/360.-p
            DH.append(H-u);Dphi.append((angle-angle0+.5)%1.)
        if count>20 or len(DH)>30:break
        count+=1
    if not DH:raise ValueError('Origin heuristic has no symmetry-phase terms')
    X,Y,Z=np.meshgrid(*[np.linspace(0,1,n) for n in shape])
    points=np.array(list(zip(X.flatten(),Y.flatten(),Z.flatten())))
    error=(points@np.asarray(DH).T+.5)%1.-np.asarray(Dphi)
    score=np.reshape(np.sum(error**2,axis=1),shape)/len(DH)
    shift=-np.array(np.unravel_index(np.argmin(score),score.shape))
    return shift,float(np.min(score)),len(DH)
