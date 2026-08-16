# NUFFT primitives + checkerboard

```python
def checkerboard(x, axes, xp):
    for ax in axes:
        s = x.shape[ax]
        phase = xp.ones(s, dtype=x.real.dtype)
        phase[1::2] = -1
        view_shape = [1] * x.ndim
        view_shape[ax] = s
        x = x * phase.reshape(view_shape)
    return x

def nufft_forward(u, xi, n, xp):
    U = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(u)))
    grid_coords = (xi + 0.5) * n
    idx = xp.clip(xp.round(grid_coords).astype(xp.int32), 0, n - 1)
    return U[idx[:, 0], idx[:, 1], idx[:, 2]]

def nufft_adjoint(F, xi, n, xp):
    grid_coords = (xi + 0.5) * n
    idx = xp.clip(xp.round(grid_coords).astype(xp.int32), 0, n - 1)
    U_real = xp.zeros((n, n, n), dtype=xp.float32)
    U_imag = xp.zeros((n, n, n), dtype=xp.float32)
    indices = (idx[:, 0], idx[:, 1], idx[:, 2])
    if xp.__name__ == 'numpy':
        xp.add.at(U_real, indices, F.real.astype(xp.float32))
        xp.add.at(U_imag, indices, F.imag.astype(xp.float32))
    else:
        import cupyx
        cupyx.scatter_add(U_real, indices, F.real.astype(xp.float32))
        cupyx.scatter_add(U_imag, indices, F.imag.astype(xp.float32))
    U = (U_real + 1j * U_imag).astype(xp.complex64)
    return xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(U)))
```
