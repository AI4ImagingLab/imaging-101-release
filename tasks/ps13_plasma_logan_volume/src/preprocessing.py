"""Calibrated dynamic PET and measured arterial input; no imaging-library imports."""
from pathlib import Path
import numpy as np
SECONDS_PER_MINUTE=60.0
LATE_START_MINUTES=30.0

def parent_input(blood_time,plasma_activity,parent_fraction,frame_start,frame_duration):
    """Return parent-plasma concentration at frame midpoints, Bq/mL.
    All input times are seconds. Append known preinjection zero at t=0,
    linearly interpolate measured parent concentrations, no decay reapplication.
    """
    mid=frame_start/SECONDS_PER_MINUTE+.5*(frame_duration/SECONDS_PER_MINUTE)
    return np.interp(mid,np.r_[0.,blood_time/SECONDS_PER_MINUTE],np.r_[0.,plasma_activity*parent_fraction])

def load_data(path="data/raw_data.npz"):
    """Return activity(z,y,x,t)float64,mask(z,y,x)bool,input(T),start(T),duration(T)minutes,metadata dict."""
    with np.load(Path(path),allow_pickle=False) as z:data={k:z[k] for k in z.files}
    shapes={"activity":(1,27,1,200,200),"frame_start":(1,27),"frame_duration":(1,27),"blood_time":(1,19),"plasma_activity":(1,19),"parent_fraction":(1,19),"valid_mask":(1,1,200,200),"voxel_to_world":(1,4,4)}
    if set(data)!=set(shapes):raise ValueError("Invalid observation/calibration keys")
    for k,shape in shapes.items():
        expected=np.uint8 if k=="valid_mask" else np.float32 if k=="activity" else np.float64
        if data[k].shape!=shape or data[k].dtype!=expected or not np.isfinite(data[k]).all():raise ValueError(f"Invalid {k} shape/dtype/finiteness")
    if not np.isin(data["valid_mask"],[0,1]).all():raise ValueError("Mask must be binary")
    start=data["frame_start"][0];duration=data["frame_duration"][0];blood=data["blood_time"][0]
    if not np.all(np.diff(start)>0) or not np.all(duration>0) or not np.all(np.diff(blood)>0):raise ValueError("Invalid timing")
    if not np.all((data["parent_fraction"]>=0)&(data["parent_fraction"]<=1)):raise ValueError("Invalid parent fraction")
    cp=parent_input(blood,data["plasma_activity"][0],data["parent_fraction"][0],start,duration)
    activity=np.moveaxis(data["activity"][0],0,-1).astype(np.float64);mask=data["valid_mask"][0].astype(bool)
    if not np.all(activity[mask][:,start>=1800]>0):raise ValueError("Late tissue concentrations must be positive")
    return activity,mask,cp,start/SECONDS_PER_MINUTE,duration/SECONDS_PER_MINUTE,data
