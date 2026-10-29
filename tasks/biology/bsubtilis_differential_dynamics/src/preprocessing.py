"""Local DDM preprocessing extracted from fastddm (GPL-3.0-or-later).
Copyright2023-present University of Vienna; adapted2026 to explicit arrays.
"""
import numpy as np
from scipy import fft as spfft
from scipy.interpolate import interp1d
PIXEL_SIZE_UM=75.59/128
FRAME_RATE_HZ=100.
Q_MIN=.75
Q_MAX=4.

def load_observations(path):
    with np.load(path,allow_pickle=False) as z:frames=z['frames'].copy()
    if frames.shape!=(1,2,2000,128,128) or frames.dtype!=np.uint8:raise ValueError('Expected full paired native movies')
    return frames

def logspace_int(stop,num=200,endpoint=False):
    samples=np.empty(num);ratio=stop**(1./(num if endpoint else num+1))
    if not endpoint:num+=1
    samples[0]=1
    for i in range(1,len(samples)):
        nextval=samples[i-1]*ratio
        if nextval-samples[i-1]>=1.:samples[i]=nextval
        else:
            samples[i]=samples[i-1]+1
            ratio=(stop/samples[i])**(1/(num-i-1))
    return np.round(samples).astype(int)

def image_structure(movie,chunk_size=256):
    """Full (T,H,W) native movie -> packed (T+1,H,W//2+1) ADU².
    Lags1..T-1 followed by mean power and temporal Fourier variance.
    Spatial FFT is normalized by sqrt(HW), y-shifted only. Temporal FFT
    zero-pads to2T and computes unbiased finite-overlap autocorrelation.
    Frequency columns are independently chunked to bound memory; temporal
    sums and FFTs retain the original order for each column, workers1.
    """
    t,h,w=movie.shape
    spectrum=spfft.rfft2(movie.astype(np.float64),s=(h,w),workers=1)/np.sqrt(h*w)
    flat=spectrum.reshape(t,-1)
    packed=np.empty((t+1,flat.shape[1]),dtype=np.float64)
    lag=np.arange(1,t)
    for start in range(0,flat.shape[1],chunk_size):
        end=min(start+chunk_size,flat.shape[1]);z=flat[:,start:end]
        sq=np.abs(z)**2
        ft=spfft.fft(z.astype(np.complex128),n=2*t,axis=0,workers=1)
        ac=spfft.ifft(np.abs(ft)**2,axis=0,workers=1)[:t].real
        cs=np.cumsum(sq.astype(np.float64),axis=0)
        rev=np.cumsum(sq[::-1].astype(np.float64),axis=0)
        packed[:-2,start:end]=(cs[-lag-1]+rev[-lag-1]-2*ac[lag])/(t-lag)[:,None]
        packed[-2,start:end]=sq.mean(axis=0)
        packed[-1,start:end]=z.var(axis=0)
    return spfft.fftshift(packed.reshape(t+1,h,w//2+1),axes=-2)

def radial_average(packed,pixel_size=PIXEL_SIZE_UM):
    """Return averages/SD (q,lag+2), q rad/um, retaining original counts.
    Full-plane multiplicities are2 except DC/Nyquist columns1. Remove
    central row and first column. Original90-edge searchsorted convention
    and loop accumulation order are preserved; empty bins contain NaN.
    SD is unbiased spatial weighted dispersion, not temporal fit uncertainty.
    """
    _,h,half=packed.shape;w=(half-1)*2
    kx=2*np.pi*np.fft.fftfreq(w)[:half];kx*=1./pixel_size
    ky=2*np.pi*np.fft.fftshift(np.fft.fftfreq(h));ky*=1./pixel_size
    xx,yy=np.meshgrid(kx,ky);dist=np.sqrt(xx**2+yy**2)
    bins=int(np.sqrt(2)*(half-1));edges=np.linspace(0,kx[1]*bins,bins,dtype=np.float64)
    ids=np.searchsorted(edges,dist);mask=np.ones(dist.shape,dtype=bool);mask[:,0]=False;mask[h//2]=False;mask[(ids==bins)|(dist<0)]=False
    counts=np.full(dist.shape,2,dtype=np.float64);counts[:,0]=1;counts[:,-1]=1
    sumw=np.zeros(bins);sumw2=np.zeros(bins)
    for (i,j),index in np.ndenumerate(ids):
        if mask[i,j]:sumw[index]+=counts[i,j];sumw2[index]+=counts[i,j]
    q=np.zeros(bins);avg=np.zeros((bins,len(packed)));err=np.zeros_like(avg)
    for (i,j),index in np.ndenumerate(ids):
        if mask[i,j]:
            q[index]+=dist[i,j]*counts[i,j]/sumw[index]
            avg[index]+=packed[:,i,j]*counts[i,j]/sumw[index]
    avg[sumw==0]=np.nan
    for index in range(bins):
        if sumw[index]==0:q[index]=.5*(edges[index-1]+edges[index]) if index>0 else edges[0]
    bias=np.array([float(a)-float(c)/float(a) if a>0 else 0 for a,c in zip(sumw,sumw2)])
    with np.errstate(divide='ignore',invalid='ignore'):
        for (i,j),index in np.ndenumerate(ids):
            if mask[i,j]:err[index]+=counts[i,j]*(packed[:,i,j]-avg[index])**2/bias[index]
    err[sumw==0]=np.nan
    return avg,np.sqrt(err),q

def resample_loglog(averages,errors,frame_rate=FRAME_RATE_HZ,num=200):
    """Quadratic log–log interpolation onto original quasi-log integer lags."""
    tau=np.arange(1,averages.shape[-1]-1,dtype=np.float64)*(1./frame_rate)
    newtau=logspace_int(len(tau)//2,num=num)*tau[0]
    out=np.full((len(averages),len(newtau)),np.nan);out_err=np.full_like(out,np.nan)
    for i in range(len(averages)):
        if np.isnan(averages[i,0]):continue
        for dest,source in ((out,averages),(out_err,errors)):
            f=interp1d(np.log(tau),np.log(source[i,:-2]),kind='quadratic',fill_value='extrapolate')
            dest[i]=np.exp(f(np.log(newtau)))
    return out,out_err,newtau
