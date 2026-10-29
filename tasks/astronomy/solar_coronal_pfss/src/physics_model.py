"""Finite-volume spherical magnetic flux and potential-to-field operators.
Adapted from pfsspy, Yeates2017 and contributors2018 onward; GPL3 LICENSE.
"""
import numpy as np
def grid(ns=180,nphi=360,nr=35,rss=2.5):
    """Coordinates in log-radius, sine-latitude and longitude (radians).
    Dimensionless solar-radius units; phi=0 corresponds to observed map edge.
    """
    ds=2/ns;dp=2*np.pi/nphi;dr=np.log(rss)/nr
    return dict(ns=ns,nphi=nphi,nr=nr,rss=rss,ds=ds,dp=dp,dr=dr,sg=np.linspace(-1,1,ns+1),sc=np.linspace(-1+ds/2,1-ds/2,ns),rg=np.linspace(0,np.log(rss),nr+1),rc=np.linspace(dr/2,np.log(rss)-dr/2,nr),pg=np.linspace(0,2*np.pi,nphi+1))
def area_factors(g):
    """Return radial/polar/azimuthal cell-face areas in solar-radius squared."""
    dr,ds,dp=g['dr'],g['ds'],g['dp'];sg=g['sg'];rg=g['rg'];rc=g['rc']
    ar=np.exp(2*rg)[:,None]*np.ones((1,g['ns']))*ds*dp
    at=.5*np.exp(2*rc-dr)[:,None]*(np.exp(2*dr)-1)*dp*np.sqrt(1-sg**2)[None]
    at[:,0]=at[:,1];at[:,-1]=at[:,-2]
    ap=.5*np.exp(2*rc-dr)[:,None]*(np.exp(2*dr)-1)*(np.arcsin(sg[1:])-np.arcsin(sg[:-1]))[None]
    return ar,at,ap
def field_from_potential(als,alp,g):
    """Curl edge-integrated A to staggered Br,Btheta,Bphi, all Gauss.
    als:(nphi+1,ns,nr+1), alp:(nphi,ns+1,nr+1).
    Returns r-first fields without interpolating non-collocated components.
    """
    ar,at,ap=area_factors(g);ns,np_,nr=g['ns'],g['nphi'],g['nr']
    fr=als[1:]-als[:-1]+alp[:,:-1]-alp[:,1:]
    fs=alp[:,:,1:]-alp[:,:,:-1]
    fp=als[:,:,:-1]-als[:,:,1:]
    # Polar ghost convention of the original DuMFriC discretization.
    for i in range(np_):
        # Original padded-phi indexing has physical rows i+1 and opposite
        # padded row (i+1+nphi/2)%nphi; translate it exactly.
        opposite=(i+np_//2)%np_
        fs[i,0]=.5*(fs[i,1]-fs[opposite,1])
        fs[i,-1]=.5*(fs[i,-2]-fs[opposite,-2])
    return dict(radial=fr.transpose(2,1,0)/ar[:,:,None],polar=-fs.transpose(2,1,0)/at[:,:,None],azimuthal=fp.transpose(2,1,0)/ap[:,:,None])
def flux_divergence(fields,g):
    """Net outward cell flux, Gauss*Rsun²; polar sign uses southward theta."""
    ar,at,ap=area_factors(g)
    r=fields['radial']*ar[:,:,None];t=fields['polar']*at[:,:,None];p=fields['azimuthal']*ap[:,:,None]
    # The true polar face has zero area. Adjacent-face surrogate areas above
    # recover a regular field value at the coordinate singularity; they must
    # never be interpreted as physical finite polar surface areas.
    t[:,0]=0;t[:,-1]=0
    return r[1:]-r[:-1]-(t[:,1:]-t[:,:-1])+(p[:,:,1:]-p[:,:,:-1])
