# Latent QP formula

For latent variable `Z[t]`:

```text
J = (Z[0]-z_b)^T B (Z[0]-z_b)
  + sum_t (Z[t]-seq_z[t])^T R (Z[t]-seq_z[t])
  + sum_{t=1}^{T-1} penalty[t-1] * (Z[t] - F^T Z[t-1])^T Q (Z[t] - F^T Z[t-1])
```

For `T=5`:

```python
penalty = np.linspace(0.1, 1, 4)[::-1]  # array([1.0, 0.7, 0.4, 0.1])
```

The matrices `B`, `R`, and `Q` are already precision matrices. Do not invert them. Do not add box constraints or normalization constraints.
