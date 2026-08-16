# Forward model detail

For an object `X[z, y, x]` and PSF `h[z, py, px, c]`, the estimated data is:

```text
Yhat[y, x, c] = sum_z conv2_same(X[z], h[z, :, :, c])
```

The inverse update needs the matched adjoint/backprojection:

```text
B[z, y, x] = sum_c conv2_same(R[:, :, c], flip_spatial(h[z, :, :, c]))
```

where `R = Y / Yhat` with a numerical guard for small `Yhat`.

Important axis conventions:

- Object: `(z, y, x)`.
- Measurement: `(y, x, channel)`.
- PSF: `(z, y, x, channel)`.
- Spatial flips are over PSF row/column/channel axes in the FFT implementation described later; the effective adjoint must reverse the spatial convolution kernel for the ratio backprojection.
