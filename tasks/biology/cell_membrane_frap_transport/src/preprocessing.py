"""ImageJ-compatible sampling and author HiFRAP normalization.

HiFRAP adaptations: Lorenzetti et al., CC BY-NC-SA 4.0. See NOTICE.
ImageJ sampling/downsizing formulas: public-domain ImageJ 1.54f.
"""
import numpy as np
from scipy.optimize import curve_fit

PIXEL_SIZE = np.float32(0.0431)
TIME_STEP = np.float32(30)

def sample_line(movie, coordinates):
    """Sample uint16 (time,y,x) at float32 (position,xy) strip centers.

    Nine transverse bilinear samples are rounded to float32, then reduced
    with the original triangular kernel and replicated strip boundaries.
    Returns float32 (time,position), in original camera units.
    """
    xy=np.asarray(coordinates,dtype=np.float32)
    prev=np.vstack((xy[0]-(xy[1]-xy[0]),xy[:-1]))
    dx=xy[:,0].astype(float)-prev[:,0]
    dy=prev[:,1].astype(float)-xy[:,1]
    length=np.sqrt(dx*dx+dy*dy).astype(np.float32)
    dx/=length; dy/=length
    x=xy[:,0]+np.arange(-4,5)[:,None]*dy
    y=xy[:,1]+np.arange(-4,5)[:,None]*dx
    xi=np.floor(x).astype(int); yi=np.floor(y).astype(int)
    if np.any(xi<0) or np.any(yi<0) or np.any(xi+1>=movie.shape[2]) or np.any(yi+1>=movie.shape[1]):
        raise ValueError('Sampling strip exceeds movie boundaries')
    fx=x-xi; fy=y-yi
    strips=((1-fy)*((1-fx)*movie[:,yi,xi]+fx*movie[:,yi,xi+1])+fy*((1-fx)*movie[:,yi+1,xi]+fx*movie[:,yi+1,xi+1])).astype(np.float32)
    weights=np.zeros(9,dtype=np.float32); total=0.
    for src in range(-4,13):
        weight=np.float32(1)-np.float32(abs((src+.5)/9-.5))
        total+=float(weight)
        weights[np.clip(src,0,8)]+=weight
    weights=(weights/total).astype(np.float32)
    output=np.zeros((len(movie),len(xy)),dtype=np.float32)
    for j in range(9): output+=strips[:,j]*weights[j]
    return output

def prepare(data):
    """Use all movie frames; return raw profiles and normalized observations."""
    movie=data['movie'][0]
    profile=sample_line(movie,data['frap_coordinates'][0])
    control=sample_line(movie,data['control_coordinates'][0])
    background=movie[:,50:452,63:204].astype(np.float32)
    shift=np.min(profile)*1.2
    amplitude=(np.max(profile)-shift)*.8
    y=(profile-shift)/amplitude
    yc=(control-shift)/amplitude
    bg=np.mean((background-shift)/amplitude)
    stationary=np.mean(y[:1])-bg
    control_mean=np.mean(yc,axis=1)
    def exponential(t,a,b): return a*np.exp(-b*t)+bg
    fit,_=curve_fit(exponential,np.arange(42),control_mean,p0=[1,.001],bounds=([0,0],[1000,1/42*1000]))
    return dict(profile=profile,control=control,normalized=y[1:],background=bg,
                stationary=stationary,epsilon=fit[1],norm=np.array([shift,amplitude]),
                control_fit=exponential(np.arange(42),*fit)*amplitude+shift)
