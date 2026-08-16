# Latent constants

```python
D = 512
B = np.eye(D) * 0.1
R = np.eye(D)
Q = np.eye(D) * 0.1
```

The dynamics matrix `F` is the loaded `C_forward` tensor copied to CPU NumPy. In the QP, model consistency compares `z_t` to `F.T @ z_{t-1}`.

The background vector `z_b` is the loaded latent mean, copied to CPU NumPy with shape `(512,)` before QP construction.
