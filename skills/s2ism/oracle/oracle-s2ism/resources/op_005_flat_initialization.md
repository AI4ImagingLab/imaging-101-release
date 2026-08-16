# Initialization note

Use:

```python
Nz = h.shape[0]
Ny, Nx = y.shape[:2]
X = np.ones((Nz, Ny, Nx), dtype=np.float32) * (y.sum() / (Nz * Ny * Nx))
```

Avoid zeros in `X` because a multiplicative Richardson-Lucy/ML-EM update cannot recover a voxel that starts at exactly zero. A tiny floor can be used if an alternative initialization creates zeros.
