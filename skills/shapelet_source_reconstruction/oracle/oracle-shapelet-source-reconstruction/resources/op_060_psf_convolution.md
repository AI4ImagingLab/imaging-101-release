Gaussian PSF (scipy).

```python
import numpy as np
import scipy.ndimage

def fwhm2sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def gaussian_convolve(image, fwhm, pixel_size, truncation=5):
    if fwhm <= 0:
        return image.copy()
    sigma_pix = fwhm2sigma(fwhm) / pixel_size
    return scipy.ndimage.gaussian_filter(image, sigma_pix, mode='constant', truncate=truncation)
```

Key points: pixel_size is the supersampled pixel size (deltaPix/supersampling). truncate=5 in the forward pipeline and in response-matrix column construction. For fwhm_lensing=0.1 and pixel_size=0.01, sigma_pix ~= 4.247.