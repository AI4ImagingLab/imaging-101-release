"""GPL-3.0, adapted from PyCCAPT authors: explicit evaporation/projection mathematics."""
import numpy as np
FLIGHT_PATH_MM=110.;FIELD_FACTOR=4.;DETECTION_EFFICIENCY=.7;IMAGE_COMPRESSION=1.4;EVAPORATION_FIELD_V_NM=19.;ATOMIC_DENSITY_NM3=60.2
def radius_evolution(voltage,kf=FIELD_FACTOR,field_evap=EVAPORATION_FIELD_V_NM):
    """Specimen radius R=V/(kf*F), float64(N,), nm; F in V/nm."""
    if kf<=0 or field_evap<=0:raise ValueError('Positive field parameters required')
    return np.asarray(voltage,dtype=np.float64)/(kf*field_evap)
def launch_geometry(xy,voltage,flight_path_mm=FLIGHT_PATH_MM,kf=FIELD_FACTOR,icf=IMAGE_COMPRESSION,field_evap=EVAPORATION_FIELD_V_NM):
    """Return radius nm, azimuth rad, corrected launch angle rad and detector radius mm."""
    if flight_path_mm<=0 or icf<=0:raise ValueError('Positive projection parameters required')
    rad=np.hypot(xy[:,0]*10,xy[:,1]*10);phi=np.arctan2(xy[:,1]*10,xy[:,0]*10)
    theta_p=np.arctan(rad/flight_path_mm);argument=(icf-1)*np.sin(theta_p)
    if np.max(np.abs(argument))>1:raise ValueError('Image compression outside launch geometry domain')
    return radius_evolution(voltage,kf,field_evap),phi,theta_p+np.arcsin(argument),rad
def depth_increments(xy,voltage,flight_path_mm=FLIGHT_PATH_MM,kf=FIELD_FACTOR,det_eff=DETECTION_EFFICIENCY,icf=IMAGE_COMPRESSION,field_evap=EVAPORATION_FIELD_V_NM,avg_dens=ATOMIC_DENSITY_NM3):
    """Per detected-ion depth increment nm from atomic-volume conservation.

    Detector area is pi*maximum observed detector radius² in mm². Magnification
    has units mm/nm, yielding the specimen analysis area in nm². Include every ion.
    """
    if not 0<det_eff<=1 or avg_dens<=0:raise ValueError('Invalid density/efficiency')
    radius=radius_evolution(voltage,kf,field_evap);rad=np.hypot(xy[:,0]*10,xy[:,1]*10);det_area=np.max(rad)**2*np.pi
    if not det_area>0:raise ValueError('Nonzero detector field required')
    magnification=flight_path_mm/(icf*radius);specimen_area=det_area/magnification**2
    return (1/avg_dens/det_eff)/specimen_area
def project_to_detector(xyz,voltage,flight_path_mm=FLIGHT_PATH_MM,kf=FIELD_FACTOR,icf=IMAGE_COMPRESSION,field_evap=EVAPORATION_FIELD_V_NM):
    """Independent inverse of the Geiser lateral map; float64(N,2) detector cm.

    tan(theta_p)=sin(theta_a)/(cos(theta_a)+icf-1), derived from the launch-angle
    equation. z includes evaporation history and does not alter the detector ray.
    """
    radius=radius_evolution(voltage,kf,field_evap);d=np.hypot(xyz[:,0],xyz[:,1]);sine=d/radius
    if np.max(sine)>1+1e-12:raise ValueError('Position outside specimen cap')
    theta=np.arcsin(np.minimum(sine,1));rad=flight_path_mm*np.sin(theta)/(np.cos(theta)+icf-1)
    phi=np.arctan2(xyz[:,1],xyz[:,0]);return np.stack([rad*np.cos(phi)/10,rad*np.sin(phi)/10],axis=-1)
