# Update rule detail

Let `eps = torch.finfo(torch.float).eps` or a similar small float32 epsilon.

```python
Yhat = forward_project(X, h_fft)
R = where(Yhat < eps, 0, Y / Yhat)
U = back_project(R, h_adj_fft)
X_next = X * U
```

Implementation notes:

- `Y` has shape `(Ny, Nx, Nch)`.
- `X` and `X_next` have shape `(Nz, Ny, Nx)`.
- `U` is the same shape as `X`.
- Keep tensors floating point; integer data should be converted before ratios.
- If minor numerical negatives appear from FFT roundoff, clipping `U` and `X_next` to non-negative can stabilize an independent implementation, but the standard multiplicative formula generally remains non-negative.
