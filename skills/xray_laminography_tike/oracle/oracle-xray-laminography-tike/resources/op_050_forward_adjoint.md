# Forward / adjoint (depends on geometry + nufft primitives)

```python
def forward_project(obj, theta, tilt, xp, make_frequency_grid_fast, nufft_forward, checkerboard):
    obj = xp.asarray(obj, dtype=xp.complex64)
    theta = xp.asarray(theta, dtype=xp.float32)
    n = obj.shape[0]; R = theta.shape[0]
    xi = make_frequency_grid_fast(theta, tilt, n, xp)
    F = nufft_forward(obj, xi, n, xp).reshape(R, n, n)
    F = checkerboard(F, axes=(1, 2), xp=xp)
    data = xp.fft.ifft2(F, axes=(1, 2), norm='ortho')
    data = checkerboard(data, axes=(1, 2), xp=xp)
    return data.astype(xp.complex64)

def adjoint_project(data, theta, tilt, n, xp, make_frequency_grid_fast, nufft_adjoint, checkerboard):
    data = xp.asarray(data, dtype=xp.complex64)
    theta = xp.asarray(theta, dtype=xp.float32)
    F = checkerboard(data.copy(), axes=(1, 2), xp=xp)
    F = xp.fft.fft2(F, axes=(1, 2), norm='ortho')
    F = checkerboard(F, axes=(1, 2), xp=xp)
    xi = make_frequency_grid_fast(theta, tilt, n, xp)
    obj = nufft_adjoint(F.ravel(), -xi, n, xp)
    obj /= n ** 2
    return obj.astype(xp.complex64)
```
