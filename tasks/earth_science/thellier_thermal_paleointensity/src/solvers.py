"""Published SPD signed standard-major-axis inverse, with explicit uncertainty.

PmagPy SPD conventions, Lori Jonestrask / Ron Shaar; BSD-3-Clause.
This closed-form inverse has no artificial optimization iteration.
"""
import numpy as np
from .preprocessing import arai_observations,check_observations
from .physics_model import principal_direction,vector_fraction,forward_arai,scatter_box

def major_axis_fit(x,y,lab_field):
    """Return signed SMA slope, intercept, field in microtesla and uncertainty."""
    x=np.asarray(x);y=np.asarray(y);n=len(x)
    if n<3:raise ValueError('At least three paired temperatures are required')
    dx=x-np.mean(x);dy=y-np.mean(y)
    slope=np.sign(np.sum(dx*dy))*np.std(y,ddof=1)/np.std(x,ddof=1)
    if slope==0:slope=1e-10
    sigma=np.sqrt((2*np.sum(dy**2)-2*slope*np.sum(dx*dy))/((n-2)*np.sum(dx**2)))
    if sigma==0:sigma=1e-10
    intercept=np.mean(y)-slope*np.mean(x)
    return dict(slope=float(slope),sigma_slope=float(sigma),beta=float(abs(sigma/slope)),intercept=float(intercept),field=float(abs(slope)*lab_field),sigma_field=float(sigma*lab_field))

def reconstruct(raw,trim=0):
    """Reconstruct all five author-fixed intervals. trim is an explicit ablation."""
    fields=[];diagnostics=[]
    for index in range(raw['count'].shape[1]):
        a=arai_observations(raw,index);lo,hi=raw['fit_temperature'][0,index]
        keep=np.flatnonzero((a['temperature']>=lo)&(a['temperature']<=hi))
        if trim:keep=keep[trim:-trim]
        fit=major_axis_fit(a['x'][keep],a['y'][keep],raw['laboratory_field'][0,index])
        direction,mad,eigenvalues=principal_direction(a['vectors'][keep],True)
        free_direction,free_mad,_=principal_direction(a['vectors'][keep],False)
        center=np.mean(a['vectors'][keep],axis=0)
        dang=float(np.degrees(np.arccos(np.clip(np.dot(free_direction,center)/np.linalg.norm(center),-1.,1.))))
        checks=check_observations(a)
        scat,low,high,points=scatter_box(a['x'][keep],a['y'][keep],fit['slope'],checks,a['temperature'][keep[0]],a['temperature'][keep[-1]])
        fit.update(a,indices=keep,direction=direction,mad=mad,free_mad=free_mad,eigenvalues=eigenvalues,frac=vector_fraction(a['vectors'],keep[0],keep[-1]))
        fit.update(checks=checks,scat=scat,scat_low=low,scat_high=high,scat_points=points)
        fit['dang']=dang
        fit['prediction']=forward_arai(a['x'],fit['field'],raw['laboratory_field'][0,index],fit['intercept'])
        fields.append(fit['field']);diagnostics.append(fit)
    return np.array([fields],dtype=np.float64),diagnostics
