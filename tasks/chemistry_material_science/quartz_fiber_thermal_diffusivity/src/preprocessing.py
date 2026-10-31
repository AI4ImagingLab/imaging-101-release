"""Native IRBIS exports -> oriented fiber pixels. BSD3 author adaptation."""
from pathlib import Path
import numpy as np
from scipy.ndimage import rotate
PIXEL_MM=.0013
FREQUENCIES=np.array([7.547,6.154])
def load_data(path='data/raw_data.npz'):
    """Return amplitude(mK),phase(degree), both (2,164,1280) float64."""
    with np.load(Path(path),allow_pickle=False) as z:a,p=z['amplitude'],z['phase']
    if a.shape!=(1,2,164,1280) or p.shape!=a.shape or a.dtype!=np.float64 or p.dtype!=np.float64:raise ValueError('Expected float64 native arrays (1,2,164,1280)')
    if not np.isfinite(a).all() or not np.isfinite(p).all():raise ValueError('Nonfinite observation')
    return a[0],p[0]
def native_to_vertical(amplitude,phase):
    """Clip amplitude at .001mK, phase to radians, rotate90deg counterclockwise."""
    return np.rot90(np.maximum(amplitude,.001)),np.rot90(phase*np.pi/180)
def linear_fit(x,y):
    """Unweighted OLS slope,intercept,slope standard error; ignores nonfinites."""
    good=np.isfinite(x)&np.isfinite(y);x,y=np.asarray(x)[good],np.asarray(y)[good]
    if len(x)<3:raise ValueError('At least3 finite samples needed')
    xx=x-x.mean();yy=y-y.mean();ss=xx@xx
    slope=(xx@yy)/ss;intercept=y.mean()-slope*x.mean()
    error=np.sqrt(np.sum((y-(slope*x+intercept))**2)/(len(x)-2)/ss)
    return np.array([slope,intercept]),float(error)
def estimate_angle(vertical_amplitude):
    """Original bright-ridge slope refinement from fixed approximate vertical ROI."""
    # The author GUI selection is x=.1066mm, y=.2..1.4mm after native90deg rotation.
    xcenter=int(.1066/PIXEL_MM)
    crop=vertical_amplitude[:,xcenter-20:xcenter+21]
    maximum=np.nanmax(crop,axis=1);mean=np.nanmean(crop,axis=1)
    x=[];y=[]
    for i,row in enumerate(crop):
        inds=np.flatnonzero((row>=.7*maximum[i])&(row>1.5*mean[i]))
        if len(inds):x.append(inds.mean());y.append(i)
    slope=linear_fit(np.array(y),np.array(x))[0][0]
    return float(-np.rad2deg(np.arctan(slope)))
def excitation_center(image,npoints=100):
    """Mean coordinates of100largest finite amplitudes, nearest integer; (x,y)."""
    flat=image.ravel();nans=np.isnan(flat).sum()
    indices=np.argpartition(flat,-npoints-nans)[-npoints-nans:]
    indices=indices[~np.isnan(flat[indices])]
    yy,xx=np.unravel_index(indices,image.shape)
    return np.round([xx.mean(),yy.mean()]).astype(int)
def orient_and_crop(amplitude,phase,angle):
    """Order1 scipy rotation, NaN exterior, then41-column observed hot-center ROI."""
    aa=rotate(amplitude,angle,cval=np.nan,order=1);pp=rotate(phase,angle,cval=np.nan,order=1)
    x,y=excitation_center(aa);width=min(x,20,aa.shape[1]-x)
    return aa[:,x-width:x+width+1],pp[:,x-width:x+width+1],np.array([width,y])
def centerline_profiles(amplitude,phase,center,max_distance=.6):
    """Return radial distance and top/bottom central-column logamp/phase profiles."""
    x,y=map(int,center);requested=int(round(2*max_distance/PIXEL_MM))
    upper=requested;lower=requested;h=amplitude.shape[0]
    if y+.5*requested>h:upper=h-y-1
    if y-.5*requested<0:lower=y-1
    if y+upper>h:upper=h-y-1
    if y-lower<0:lower=y
    length=max(upper,lower)+1;profiles=np.full((2,length,2),np.nan)
    for j,count in enumerate((upper,lower)):
        idx=y+np.arange(count+1)*(1 if j==0 else -1)
        profiles[j,:count+1,0]=np.log(amplitude[idx,x]);profiles[j,:count+1,1]=phase[idx,x]
    return np.arange(length)*PIXEL_MM,profiles
