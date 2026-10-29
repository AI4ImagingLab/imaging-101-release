"""Spatial and spectral plotting helpers; importing does not alter a backend."""
import numpy as np
def surface_panel(axis,theta,phi,value,title,cmap='viridis',symmetric=False):
    scale=np.max(np.abs(value)) if symmetric else None
    image=axis.scatter(np.rad2deg(phi),90-np.rad2deg(theta),c=value,s=5,cmap=cmap,vmin=-scale if symmetric else None,vmax=scale)
    axis.set(xlim=(0,360),ylim=(-90,90),xlabel='Longitude (deg)',ylabel='Latitude (deg)',title=title)
    return image
def residual_panel(axis,velocity,residual,title):
    image=axis.imshow(residual,aspect='auto',origin='lower',extent=[velocity[0],velocity[-1],-.5,residual.shape[0]-.5],cmap='RdBu_r')
    axis.set(xlabel='Observed velocity (km/s)',ylabel='Observation index',title=title)
    return image
