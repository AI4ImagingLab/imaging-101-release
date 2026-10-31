"""Explicit collinear polarized layer reflection and instrumental resolution.
Refl1D mathematical conventions, see LICENSE and validation provenance.
"""
import numpy as np
from scipy.special import erf
SUBSTRATE_SLD=5.377

def unpack(parameters,dead=True):
    if dead:
        intensity,broadening,offset,rough_bottom,rough_top,dead_depth,magnetic,nuclear,thickness=parameters
    else:
        intensity,broadening,offset,rough_bottom,rough_top,magnetic,nuclear,thickness=parameters;dead_depth=0.
    return intensity,broadening,offset,rough_bottom,rough_top,dead_depth,magnetic,nuclear,thickness

def layer_stack(parameters,dead=True):
    _,_,_,sb,st,d,m,rho,h=unpack(parameters,dead)
    if dead:return np.array([0,h-d,d,0.]),np.array([0,rho,rho,SUBSTRATE_SLD]),np.array([0,m,0,0.]),np.array([st,sb,sb])
    return np.array([0,h,0.]),np.array([0,rho,SUBSTRATE_SLD]),np.array([0,m,0.]),np.array([st,sb])

def layer_reflectivity(q,width,nuclear,magnetic,roughness,spin):
    potential=(nuclear+spin*magnetic)*1e-6
    # Preserve original magnetic kernel's tiny positive absorption guard:
    # 1e-10 in the stored 1e-6 inverse-Angstrom-squared SLD convention.
    k=np.sqrt((np.asarray(q)[:,None]/2)**2-4*np.pi*potential[None,:]+4j*np.pi*1e-16)
    reflection=np.zeros(len(q),complex)
    for j in range(len(width)-2,-1,-1):
        a=k[:,j];b=k[:,j+1]
        interface=(a-b)/(a+b)*np.exp(-2*a*b*roughness[j]**2)
        phase=np.exp(2j*b*width[j+1]);reflection=(interface+reflection*phase)/(1+interface*reflection*phase)
    return abs(reflection)**2

def gaussian_resolution(q,values,sigma):
    # Exact integral of piecewise-linear intensity against a truncated Gaussian.
    # Whole intervals crossing the original exp(-6.907755...) cutoff are retained.
    q=np.asarray(q);values=np.asarray(values);sigma=np.asarray(sigma)
    limit=sigma*np.sqrt(2*6.90775527898213703123)
    left=np.clip(np.searchsorted(q,q-limit,side='right')-1,0,len(q)-2)
    right=np.clip(np.searchsorted(q,q+limit,side='left'),1,len(q)-1)
    x=q[None,:];center=q[:,None];s=sigma[:,None]
    errors=erf((x-center)/(np.sqrt(2)*s));gaussian=np.exp(-(x-center)**2/(2*s*s))
    slope=np.diff(values)/np.diff(q);intercept=values[1:]-slope*q[1:]
    area=.5*(slope[None,:]*center+intercept[None,:])*np.diff(errors,axis=1)-s/np.sqrt(2*np.pi)*slope[None,:]*np.diff(gaussian,axis=1)
    index=np.arange(len(q)-1)[None,:];mask=(index>=left[:,None])&(index<right[:,None])
    norm=errors[np.arange(len(q)),right]-errors[np.arange(len(q)),left]
    return 2*np.sum(np.where(mask,area,0),axis=1)/norm

def predictions(parameters,data,dead=True):
    intensity,broadening,offset,*_=unpack(parameters,dead);width,nuclear,magnetic,roughness=layer_stack(parameters,dead);result=[]
    for channel,spin in enumerate([1,-1]):
        angle=data['angle'][channel];wavelength=data['wavelength'][channel]
        q=4*np.pi/wavelength*np.sin(np.deg2rad(angle+offset))
        divergence=np.deg2rad(data['angle_resolution'][channel])/np.sqrt(8*np.log(2))
        extra=np.deg2rad(broadening)/np.sqrt(8*np.log(2))
        sigma=np.sqrt(data['q_resolution'][channel]**2+(4*np.pi/wavelength*np.cos(np.deg2rad(angle)))**2*(2*extra*divergence+extra**2))
        r=layer_reflectivity(q,width,nuclear,magnetic,roughness,spin)
        result.append(intensity*gaussian_resolution(q,r,sigma))
    return np.array(result)

def depth_profiles(parameters,depth,dead=True):
    _,_,_,sb,st,d,m,rho,h=unpack(parameters,dead)
    bottom=.5*(1+erf(depth/(np.sqrt(2)*sb)));top=.5*(1+erf((depth-h)/(np.sqrt(2)*st)))
    magnetic_bottom=.5*(1+erf((depth-d)/(np.sqrt(2)*sb)))
    nuclear=SUBSTRATE_SLD+(rho-SUBSTRATE_SLD)*bottom-rho*top
    magnetic=m*(magnetic_bottom-top)
    return np.stack([nuclear,magnetic],axis=-1)
