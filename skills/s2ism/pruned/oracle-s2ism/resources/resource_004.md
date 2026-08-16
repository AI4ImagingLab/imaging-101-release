# PSF preparation formulas

Given `data_check.shape == (Ny, Nx, Nch)` and `h.shape == (Nz, hy, hx, Nch)`:

```python
crop_pad_y = int((Ny - hy) / 2)
crop_pad_x = int((Nx - hx) / 2)
```

For the public case both values are 28. Pad `h` only along the two spatial axes, not the z or channel axes.

Normalization:

```python
h = h / h.sum(axis=(1, 2, 3), keepdims=True)
```

This is per axial plane. For the public raw PSF, the plane sums are already approximately one, but normalizing after padding avoids small drift and matches the solver convention.

Adjoint kernel:

```python
h_flip = flip(h, axes=(1, 2, 3))
```

If implementing direct convolution rather than the FFT convention, at minimum flip spatial axes `(1, 2)` for the adjoint ratio backprojection and verify the channel summation is applied correctly.
