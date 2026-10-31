"""Bound LO Peg acquisition and explicit maximum-entropy reconstruction."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from . import physics_model as physics
from . import solvers as mem
from .calibration import fitLineStrength

MODE = 'brightness'
C_KMS = 299792.458

def load_data(path='data/raw_data.npz'):
    """Load finite, batch-first continuum-normalized observations without pickle."""
    with np.load(path,allow_pickle=False) as source:
        data={k:source[k][0].copy() for k in source.files}
    for key,value in data.items():
        if not np.all(np.isfinite(value)):raise ValueError('Nonfinite '+key)
    return data

def prepare(data,mode=MODE):
    """Build the Roche surface, observing views and Voigt/weak-field profiles.

    Returns an explicit state namespace.  Its arrays are newly allocated; no
    hidden map, fitted harmonic coefficient or upstream file is loaded.
    """
    magnetic=mode=='magnetic'
    p=SimpleNamespace(fitBri=int(not magnetic),fitMag=int(magnetic),calcDI=int(not magnetic),calcDV=int(magnetic),
        fEntropyBright=1,magGeomType='full',defaultBright=1.,defaultBent=100.,maximumBright=1.01,
        brightEntScale=1.,instrumentRes=65000.,velEq=67.7/np.sin(np.pi/4),cycleList=(data['julian_date']-2456892.015)/.4232)
    grid=physics.starGrid(60,.4232,.66,.72,0)
    bright=SimpleNamespace(bright=np.ones(grid.numPoints))
    field=physics.magSphHarmoics(15);field.initMagGeom(grid.clat,grid.long)
    line=SimpleNamespace(wl0=np.array([650.]),str=np.array([.6306]),widthGauss=np.array([2.41]),widthLorentz=np.array([.89]),g=np.array([1.195]),limbDark=np.array([.66]),gravDark=np.array([.5]),instRes=65000.,numLines=1)
    cycles=physics.getCyclesClat(.4232,.02,data['julian_date'],2456892.015,grid.clat)
    views=[physics.visibleGrid(grid,np.pi/4,row,.4232,.02) for row in cycles]
    wavelength=[(v+19.8)/C_KMS*650.+650. for v in data['velocity']]
    cart=field.getAllMagVectorsCart()
    derivative=field.getAllMagDerivsCart() if magnetic else 0.
    observed_ew=0.
    for v,intensity in zip(data['velocity'],data['stokes_i']):
        value=0.
        for i in range(len(v)-1):value+=(1.-intensity[i])*(v[i+1]-v[i])/C_KMS*650.
        value+=(1.-intensity[-1])*(v[-1]-v[-2])/C_KMS*650.
        observed_ew+=value
    observed_ew/=len(views)
    fitLineStrength(observed_ew,p,views,cart,derivative,bright,line,wavelength,0)
    profiles=physics.getAllProfDiriv(p,views,cart,derivative,bright,line,wavelength)
    observation=data['stokes_v' if magnetic else 'stokes_i'].ravel().copy()
    variance=data['sigma_v' if magnetic else 'sigma_i'].ravel()**2
    constants=mem.constantsMEM(p,bright,field,observation.size)
    response=mem.packResponseMatrix(profiles,observation.size,constants.npBriMap,field,'full',p.fitBri,p.fitMag) if magnetic else None
    if magnetic:p.calcDV=0
    weights=mem.setEntropyWeights(p,field,grid)
    return SimpleNamespace(par=p,grid=grid,bright=bright,field=field,line=line,views=views,wavelength=wavelength,
        derivative=derivative,profiles=profiles,observation=observation,variance=variance,constants=constants,response=response,weights=weights)

def reconstruct(data,target=1.,max_iterations=20,test_tolerance=1e-4,mode=MODE):
    """Run the published ten-direction Skilling--Bryan updates.

    Diagnostics describe the current evaluated image; the returned proposal is
    applied at the next iteration.  Original brightness aliasing is retained:
    the brightness map is updated in place, whereas magnetic coefficients are
    unpacked only when the next iteration begins.
    """
    s=prepare(data,mode);p=s.par;c=s.constants
    history=[];images=[];first={}
    for iteration in range(max_iterations):
        if iteration:mem.unpackImageVector(image,s.bright,s.field,'full',p.fitBri,p.fitMag)
        cart=s.field.getAllMagVectorsCart()
        for view,profile in zip(s.views,s.profiles):
            profile.updateIntProfDeriv(view,cart,s.derivative,s.bright,s.line,p.calcDI,p.calcDV)
        model=mem.packModelVector(s.profiles,p.fitBri,p.fitMag)
        image=mem.packImageVector(s.bright,s.field,'full',p.fitBri,p.fitMag)
        images.append(image.copy())
        if p.calcDI or p.calcDV:
            s.response=mem.packResponseMatrix(s.profiles,s.observation.size,c.npBriMap,s.field,'full',p.calcDI,p.calcDV)
        if iteration==0:first=dict(image=image.copy(),model=model.copy(),response=s.response.copy())
        result=mem.mem_iter(c.n1Model,c.n2Model,c.nModelTot,image,s.observation,model,s.variance,s.response,s.weights,
            1.,100.,1.01,target*s.observation.size,0)
        entropy,chi2,test,image,*_=result
        history.append([entropy,chi2,test])
        if iteration==0:first.update(next_image=image.copy(),entropy=entropy,chi2=chi2,test=test)
        if chi2<=target*s.observation.size*1.001 and test<test_tolerance:break
    result=s.bright.bright.copy() if mode=='brightness' else s.field.getAllMagVectors().copy()
    s.history=np.array(history);s.iterate_images=np.array(images);s.first=first
    s.final_profile=np.stack([x.IIc if mode=='brightness' else x.VIc for x in s.profiles])
    return result,s

def project_spatial(reconstruction,state,mode=MODE):
    """Synthesize measurements directly from an arbitrary spatial output field."""
    if mode=='brightness':
        state.bright.bright=np.asarray(reconstruction).copy()
        cart=np.zeros((3,state.grid.numPoints))
    else:
        br,bt,bp=np.asarray(reconstruction);theta=state.grid.clat;phi=state.grid.long
        st,ct,sp,cp=np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)
        cart=np.array([br*st*cp+bt*ct*cp-bp*sp,br*st*sp+bt*ct*sp+bp*cp,br*ct-bt*st])
    output=[]
    for view,profile in zip(state.views,state.profiles):
        profile.updateIntProfDeriv(view,cart,0.,state.bright,state.line,0,0)
        output.append(profile.IIc.copy() if mode=='brightness' else profile.VIc.copy())
    return np.stack(output)
