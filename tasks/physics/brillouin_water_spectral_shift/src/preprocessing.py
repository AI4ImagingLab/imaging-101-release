"""Native camera observations and explicit VIPA trace sampling.
Numerical method adapted from bmlab developers, GPL-3.0; see LICENSE.
"""
from pathlib import Path
import warnings
import numpy as np
from scipy.signal import medfilt2d
from scipy.ndimage import grey_closing,label
from scipy.interpolate import RegularGridInterpolator,interp1d

def load_data(path):
    with np.load(Path(path),allow_pickle=False) as f:return {k:f[k] for k in f.files}

def find_points(images,min_height=10,min_area=20,max_distance=50):
    """Six bright calibration spots from all frames, row/column pixel coordinates."""
    img=medfilt2d(np.nanmean(images,axis=0))
    base=np.median(img);height=(np.nanmean(img[img>base+min_height])+base)/2
    q=np.arange(-10,11);footprint=q[:,None]**2+q[None,:]**2<=100
    closed=grey_closing((img>height).astype(np.uint8),footprint=footprint,mode='reflect')
    labels,count=label(closed,structure=np.ones((3,3)))
    points=[]
    for i in range(1,count+1):
        mask=labels==i
        if mask.sum()>=min_area:
            spot=img.copy();spot[~mask]=0
            row,col=np.unravel_index(np.argmax(spot),spot.shape)
            distance=abs(img.shape[0]*img.shape[1]-img.shape[1]*row-img.shape[0]*col)/np.hypot(*img.shape)
            if distance<max_distance:points.append((row,col))
    return np.asarray(points)

def interpolate_time(times,values,query):
    """Linear interpolation, nearest endpoint outside calibration times."""
    return interp1d(times,values,axis=0,bounds_error=False,fill_value=(values[0],values[-1]))(query)

def extract_spectrum(image,arc):
    """Bilinear camera interpolation and mean over the radial five-pixel strip."""
    f=RegularGridInterpolator((np.arange(image.shape[0]),np.arange(image.shape[1])),image,method='linear',bounds_error=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(f((arc[:,:,0],arc[:,:,1])),axis=1)
