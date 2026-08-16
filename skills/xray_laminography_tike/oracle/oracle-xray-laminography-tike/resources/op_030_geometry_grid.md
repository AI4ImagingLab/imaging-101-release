# Frequency grid (vectorized)

```python
import numpy as np

def make_frequency_grid_fast(theta, tilt, n, xp):
    R = theta.shape[0]
    ctilt = xp.float32(np.cos(float(tilt)))
    stilt = xp.float32(np.sin(float(tilt)))
    k = (xp.arange(n, dtype=xp.float32) - n // 2) / n
    ctheta = xp.cos(theta)   # (R,)
    stheta = xp.sin(theta)   # (R,)
    ku, kv = xp.meshgrid(k, k, indexing='xy')  # both (n, n)
    ku = ku.ravel(); kv = kv.ravel()           # (n*n,)
    xi = xp.empty((R, n * n, 3), dtype=xp.float32)
    xi[:, :, 0] = kv[None, :] * stilt
    xi[:, :, 1] = (-ku[None, :] * stheta[:, None]
                    + kv[None, :] * ctheta[:, None] * ctilt)
    xi[:, :, 2] = ( ku[None, :] * ctheta[:, None]
                    + kv[None, :] * stheta[:, None] * ctilt)
    return xi.reshape(R * n * n, 3)
```

Notes:
- `xp` is numpy or cupy.
- Coordinates are in cycles/sample, range [-0.5, 0.5).
- Row axis (`kv`) is the detector `v`/slice direction; column axis (`ku`) is detector `u`.
