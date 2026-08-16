Response matrix (uses grid/array, shapelet, lens, psf helpers).

```python
import numpy as np

def build_response_matrix(numPix, deltaPix, ss, fwhm, n_max_recon, beta_recon,
                          cx, cy, kwargs_spemd=None, kwargs_shear=None,
                          apply_lens=True):
    n_super = numPix * ss
    dpix_super = deltaPix / ss
    x, y = make_grid(n_super, dpix_super)
    if apply_lens and kwargs_spemd is not None:
        x, y = ray_shoot(x, y, kwargs_spemd, kwargs_shear)
    basis = shapelet_basis_list(x, y, n_max_recon, beta_recon, cx, cy)
    A = np.zeros((len(basis), numPix * numPix))
    for i, b1d in enumerate(basis):
        b2d = array2image(b1d)
        if fwhm > 0:
            b2d = gaussian_convolve(b2d, fwhm, dpix_super, truncation=5)
        if ss > 1:
            b2d = re_size(b2d, ss)
        b2d *= deltaPix**2
        A[i, :] = image2array(b2d)
    return A

A_lens = build_response_matrix(64, 0.05, 5, 0.1, 20, 0.15, 0.2, 0.0,
                               kwargs_spemd, kwargs_shear, apply_lens=True)
# shape (231, 4096)
```
