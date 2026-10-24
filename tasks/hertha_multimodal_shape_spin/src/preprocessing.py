"""Exact native-window observation preparation, with no fitted calibration."""
from pathlib import Path
import numpy as np

PIXEL_SCALE_ARCSEC=.009942
EPOCH_JD=2443846.0

def image_fourier(image,pixel_scale=PIXEL_SCALE_ARCSEC):
    """Nonnegative reduced image, native center phase, DC normalization/drop."""
    image=np.asarray(image,dtype=np.float64)
    if image.ndim!=2 or any(size%2 for size in image.shape) or not np.isfinite(image).all():raise ValueError('Finite even image required')
    signal=np.fft.rfft2(np.maximum(image,0));ny,nx=image.shape
    if signal[0,0].real<=0:raise ValueError('Image must have positive flux')
    yy,xx=np.indices(signal.shape);signal*=(-1.)**(xx+yy);signal/=signal[0,0].real
    fx=np.arange(nx//2+1)/(nx*pixel_scale);fy=np.fft.fftfreq(ny,d=pixel_scale)
    u,v=np.meshgrid(fx,fy)
    return signal.ravel()[1:],np.stack([u.ravel()[1:],v.ravel()[1:]],axis=1)

def load_observations(path='data/raw_data.npz'):
    """Return batch-stripped legitimate observations; no evaluation assets read."""
    with np.load(Path(path),allow_pickle=False) as archive:data={key:archive[key][0].copy() for key in archive.files}
    if data['ao_images'].shape!=(2,150,150) or data['lc_curve_index'].shape!=(923,):raise ValueError('Unexpected bound acquisition shape')
    transformed=[image_fourier(image) for image in data['ao_images']]
    data['ao_fourier']=np.stack([item[0] for item in transformed]);data['ao_frequency']=transformed[0][1]
    data['ao_psf_fourier']=np.stack([image_fourier(image)[0] for image in data['ao_psf']])
    data['lc_time']=data['lc_time'].ravel();data['lc_brightness']=data['lc_brightness'].ravel()
    return data
