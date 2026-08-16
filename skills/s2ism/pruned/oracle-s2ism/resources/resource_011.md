# Output checklist

Before saving:

```python
assert recon.shape == (2, 201, 201)
assert np.isfinite(recon).all()
assert recon.dtype in (np.float32, np.float64)
```

Recommended finalization:

```python
recon = np.asarray(recon)
if recon.shape == (1, 2, 201, 201):
    recon = recon[0]
recon = np.nan_to_num(recon, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
recon = np.maximum(recon, 0.0)
np.savez('outputs/case_000.npz', reconstruction=recon.astype(np.float32))
```

The non-negativity clamp is normally redundant for multiplicative ML-EM, but protects against independent FFT roundoff or postprocessing mistakes.
