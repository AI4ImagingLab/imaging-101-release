"""Extracted Makabe et al. ICCV2025 calibration mathematics; no upstream imports.

The original estimator uses an unconstrained spectral/grating expansion and an
explicit inverse of a row-normalized KKT system. These conventions are retained.
"""
import numpy as np
from scipy.spatial import KDTree
from .preprocessing import CAMERAS,observations,population_basis
from .physics_model import EPS,interpolation_matrix,fourier_basis,wavelength_kernel,spectral_system,predicted_observations

def peak_mapping(measured,expected,degree=3,top_k=2):
    """Least squares from the two strongest pixels perRGB to spectral-bin peaks."""
    pixels=np.hstack([np.argsort(measured[:,c])[::-1][:top_k] for c in range(3)])
    bins=np.hstack([np.argsort(expected[:,c])[::-1][:top_k] for c in range(3)])
    design=np.vstack([pixels**p for p in range(degree-1,-1,-1)]).T
    return np.linalg.lstsq(design,bins.astype(float),rcond=None)[0]

def icp_mapping(measured,expected,degree=3,iterations=500):
    """Author LED-only point-to-plane polynomial ICP, fixed500 updates, no early stop."""
    f=len(expected);F=len(measured)
    src=measured.T/measured.max(axis=0)[:,None]
    target=expected.T/expected.max(axis=0)[:,None]
    clouds=[np.column_stack((np.arange(f),target[c])) for c in range(3)]
    normals=[]
    for cloud in clouds:
        differences=np.diff(cloud,axis=0);differences=np.vstack((differences,differences[-1]))
        tangents=differences/np.linalg.norm(differences,axis=1)[:,None]
        normals.append(np.column_stack((-tangents[:,1],tangents[:,0])))
    trees=[KDTree(c) for c in clouds]
    design=np.vstack([np.arange(F)**p for p in range(degree-1,-1,-1)]).T
    weights=np.zeros(degree);weights[-2:]=[f/F,0]
    for _ in range(iterations):
        x=design@weights;A=[];b=[]
        for c in range(3):
            _,idx=trees[c].query(np.column_stack((x,src[c])),k=1)
            n=normals[c][idx]
            A.append(design*n[:,0,None]);b.append(np.sum(n*clouds[c][idx],axis=1)-n[:,1]*src[c])
        weights=np.linalg.lstsq(np.vstack(A),np.hstack(b),rcond=None)[0]
    return weights

def reconstruct_camera(data,camera,basis_size=7,attenuation_size=7,fluorescent=True):
    """Return normalized31x3 response plus full acquisition/model diagnostics."""
    direct,dispersed,flu=observations(data,camera)
    bases,mean=population_basis(data['population_sensitivity'])
    interp=interpolation_matrix(data['illumination_wavelength'],data['population_wavelength'])
    expected=data['fluorescent_spectrum'][:,None]*(interp@mean)
    if fluorescent:weights=peak_mapping(flu,expected)
    else:
        # Original initial reciprocal attenuation is a least-squares constant1.
        eta=fourier_basis(len(data['led_spectrum']),attenuation_size)
        initial=np.linalg.lstsq(eta,np.ones(len(eta)),rcond=None)[0]
        attenuation=1/(eta@initial+EPS)
        expected_led=data['led_spectrum'][:,None]*(np.clip(attenuation,0,None)[:,None]+EPS)*(interp@mean)
        weights=icp_mapping(dispersed,expected_led)
    mapping=np.polyval(weights,np.arange(len(dispersed)))
    kernel=wavelength_kernel(mapping,len(data['led_spectrum']))
    attenuation_basis=fourier_basis(len(data['led_spectrum']),attenuation_size)
    system=spectral_system(direct,dispersed,data['led_spectrum'],kernel,interp,bases[:,:,:basis_size],attenuation_basis)
    solution=np.linalg.inv(system['matrix'])@system['rhs']
    coefficient=solution[attenuation_size:-3].reshape(3,basis_size).T
    sensitivity=np.column_stack([bases[c,:,:basis_size]@coefficient[:,c] for c in range(3)])
    attenuation=1/(attenuation_basis@solution[:attenuation_size]+EPS)
    pred_direct,pred_dispersed=predicted_observations(sensitivity,attenuation,data['led_spectrum'],kernel,interp)
    normalized=sensitivity/sensitivity.max()
    return dict(reconstruction=normalized,sensitivity=sensitivity,attenuation=attenuation,weights=weights,mapping=mapping,kernel=kernel,interpolation=interp,bases=bases,system=system,solution=solution,pred_direct=pred_direct,pred_dispersed=pred_dispersed,direct=direct,dispersed=dispersed,fluorescent=flu)

def reconstruct(data,**kwargs):
    """Return batch-first(1,4,31,3) full four-camera result and per-camera diagnostics."""
    details=[reconstruct_camera(data,c,**kwargs) for c in CAMERAS]
    return np.stack([d['reconstruction'] for d in details])[None],details
