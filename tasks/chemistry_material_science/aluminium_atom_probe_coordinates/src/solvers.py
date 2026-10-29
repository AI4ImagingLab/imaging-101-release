"""GPL-3.0 PyCCAPT authors' Geiser and Bas event-by-event coordinate reconstructions.

Extracted from commit69512f6fda75ed12519a36cff6e15437e072800f, preserving the
authors' current corrected Bas cm->m conversion. Numerical code is local.
"""
import numpy as np
from .preprocessing import validate_events
from .physics_model import launch_geometry,depth_increments,FLIGHT_PATH_MM,FIELD_FACTOR,DETECTION_EFFICIENCY,IMAGE_COMPRESSION,EVAPORATION_FIELD_V_NM,ATOMIC_DENSITY_NM3
def reconstruct(xy,voltage,flight_path_mm=FLIGHT_PATH_MM,kf=FIELD_FACTOR,det_eff=DETECTION_EFFICIENCY,icf=IMAGE_COMPRESSION,field_evap=EVAPORATION_FIELD_V_NM,avg_dens=ATOMIC_DENSITY_NM3):
    """Return float64(N,3) atomic coordinates nm, axes x,y,z in recorded event order.

    Initial cumulative depth includes the first detected-ion volume increment;
    cap correction is R-R*cos(theta_a), not a fitted or recentered z origin.
    """
    xy,v=validate_events(xy,voltage);radius,phi,theta,_=launch_geometry(xy,v,flight_path_mm,kf,icf,field_evap)
    lateral=radius*np.sin(theta);x=lateral*np.cos(phi);y=lateral*np.sin(phi)
    cap=radius-radius*np.cos(theta)
    dz=depth_increments(xy,v,flight_path_mm,kf,det_eff,icf,field_evap,avg_dens)
    return np.stack([x,y,np.cumsum(dz)+cap],axis=-1)
def bas_reconstruct(xy,voltage,flight_path_mm=FLIGHT_PATH_MM,kf=FIELD_FACTOR,det_eff=DETECTION_EFFICIENCY,icf=IMAGE_COMPRESSION,field_evap=EVAPORATION_FIELD_V_NM,avg_dens=ATOMIC_DENSITY_NM3):
    """Comparison only: Bas projection and current author's corrected nm output."""
    xy,v=validate_events(xy,voltage);radius=v/(kf*(field_evap/1e-9));m=(flight_path_mm*1e-3)/(icf*radius)
    x=(xy[:,0]*1e-2)/m;y=(xy[:,1]*1e-2)/m
    rad=np.hypot(xy[:,0]*1e-2,xy[:,1]*1e-2);area=np.max(rad)**2*np.pi
    if not area>0:raise ValueError('Nonzero detector field required')
    omega=1e-9**3/avg_dens
    dz=(omega*(flight_path_mm*1e-3)**2*kf**2*(field_evap/1e-9)**2)/(area*det_eff*icf**2*v**2)
    arg=1-(x*x+y*y)/(radius*radius)
    if np.min(arg)<0:raise ValueError('Position outside Bas spherical-cap domain')
    cap=radius*(1-np.sqrt(arg));z=np.cumsum(dz)+cap
    return np.stack([x*1e9,y*1e9,z*1e9],axis=-1)
