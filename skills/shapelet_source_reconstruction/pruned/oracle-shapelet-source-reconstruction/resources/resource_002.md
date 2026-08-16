Forward simulation (uses grid/array, shapelet, lens, psf helpers).

```python
import numpy as np

def simulate_image(numPix, deltaPix, ss, fwhm, kwargs_source,
                   kwargs_spemd=None, kwargs_shear=None,
                   apply_lens=True, apply_psf=True):
    n_super = numPix * ss
    dpix_super = deltaPix / ss
    x, y = make_grid(n_super, dpix_super)
    if apply_lens and kwargs_spemd is not None:
        x, y = ray_shoot(x, y, kwargs_spemd, kwargs_shear)
    s = kwargs_source
    flux = shapelet_function(x, y, s['amp'], s['n_max'], s['beta'],
                             s['center_x'], s['center_y'])
    image = array2image(flux)
    if apply_psf and fwhm > 0:
        image = gaussian_convolve(image, fwhm, dpix_super, truncation=5)
    if ss > 1:
        image = re_size(image, ss)
    image *= deltaPix**2
    return image
```

Lensed observation setup:
```python
coeff_ngc = raw['coeff_ngc'].reshape(-1)          # length 11476, n_max=150
coeff_lens = coeff_ngc / deltaPix**2               # deltaPix=0.05
kwargs_source_lens = {'n_max': 150, 'beta': 0.06, 'amp': coeff_lens,
                      'center_x': 0.2, 'center_y': 0.0}
kwargs_spemd = {'theta_E': 0.5, 'gamma': 2.0, 'center_x': 0, 'center_y': 0,
                'e1': 0.0, 'e2': 0.0}
kwargs_shear = {'gamma1': 0.0, 'gamma2': 0.0}
image_no_noise = simulate_image(64, 0.05, 5, 0.1, kwargs_source_lens,
                                kwargs_spemd, kwargs_shear,
                                apply_lens=True, apply_psf=True)  # (64,64)
```
