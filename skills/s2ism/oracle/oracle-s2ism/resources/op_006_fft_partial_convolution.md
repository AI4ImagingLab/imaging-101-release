# FFT convention pseudocode

A NumPy-style equivalent:

```python
# h: (Nz, Ny, Nx, Nch), X: (Nz, Ny, Nx)
H = rfftn(h, s=(Ny, Nx), axes=(1, 2))
Xf = rfftn(X, s=(Ny, Nx), axes=(1, 2))
Yhat = irfftn((H * Xf[:, :, :, None]).sum(axis=0), s=(Ny, Nx), axes=(0, 1)).real
Yhat = ifftshift(Yhat, axes=(0, 1))

# h_adj: same shape as h, ratio R: (Ny, Nx, Nch)
Ha = rfftn(h_adj, s=(Ny, Nx), axes=(1, 2))
Rf = rfftn(R, s=(Ny, Nx), axes=(0, 1))
Back = irfftn((Ha * Rf[None, :, :, :]).sum(axis=3), s=(Ny, Nx), axes=(1, 2)).real
Back = ifftshift(Back, axes=(1, 2))
```

Torch `einsum` can express the same operations compactly. The key is that the forward result sums over z and retains channel, while the backprojection sums over channel and retains z.
