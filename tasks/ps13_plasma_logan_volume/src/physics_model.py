"""Explicit midpoint quadrature and arterial-input Logan transformed model."""
import numpy as np

def cumulative_integral(activity,frame_start,frame_duration,integration_type="trapz"):
    """Return last-axis integral in Bq*min/mL, preserving original conventions.
    Midpoint trapezoids start with zero at the first midpoint. Original code adds
    start_time*first_activity/2 when acquisition starts after injection; the bound
    acquisition starts at0. This differs from integrating to frame-end.
    """
    a=np.asarray(activity,np.float64);start=np.asarray(frame_start,np.float64);duration=np.asarray(frame_duration,np.float64)
    initial=start[0]*a[...,0]/2
    if integration_type=="rect":return np.cumsum(a*duration,axis=-1)+initial[...,None]
    if integration_type not in {"trapz","trapz_injection"}:raise ValueError("Unsupported integration type")
    mid=start+.5*duration;pieces=(a[...,1:]+a[...,:-1])*.5*np.diff(mid)
    result=np.concatenate([np.zeros_like(a[...,:1]),np.cumsum(pieces,axis=-1)],axis=-1)+initial[...,None]
    if integration_type=="trapz_injection":result=result+duration[0]*a[...,:1]/4
    return result

def transformed_data(activity,plasma,frame_start,frame_duration,tstar=30.0,integration_type="trapz"):
    """Return design(N,L,2),response(N,L),weights(L); fit frames start>=tstar."""
    a=np.asarray(activity,np.float64);i=cumulative_integral(a,frame_start,frame_duration,integration_type);ip=cumulative_integral(plasma,frame_start,frame_duration,integration_type)
    select=frame_start>=tstar
    if select.sum()<3 or np.any(a[...,select]<=0):raise ValueError("Insufficient positive late-time data")
    x=np.stack([np.ones_like(a[...,select]),np.broadcast_to(ip[select],a[...,select].shape)/a[...,select]],axis=-1)
    return x,i[...,select]/a[...,select],frame_duration[select]

def linear_relative_residual(volume,activity,mask,plasma,frame_start,frame_duration,tstar=30.0):
    """Observation consistency after optimally eliminating the nuisance intercept."""
    x,y,w=transformed_data(activity[mask],plasma,frame_start,frame_duration,tstar)
    v=volume[mask];offset=np.sum(w*(y-v[:,None]*x[:,:,1]),axis=-1)/w.sum()
    residual=y-offset[:,None]-v[:,None]*x[:,:,1]
    return float(np.sqrt(np.sum(w*residual**2)/np.sum(w*y**2))),offset,residual
