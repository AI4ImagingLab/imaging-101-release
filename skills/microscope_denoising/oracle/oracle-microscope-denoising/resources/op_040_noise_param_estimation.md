# Block-wise noise parameter estimation

```python
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import linregress

def estimate_noise_params(y, bg=100.0, filter_size=5):
    y = y.astype(np.float64)
    H, W = y.shape[-2], y.shape[-1]
    block = 32
    means, variances = [], []
    for i in range(0, H - block, block):
        for j in range(0, W - block, block):
            b = y[..., i:i+block, j:j+block]
            smooth = uniform_filter(b, size=filter_size)
            signal_est = np.maximum(smooth - bg, 0.0).mean()
            var_est = (b - smooth).var()
            means.append(signal_est); variances.append(var_est)
    slope, intercept, *_ = linregress(np.array(means), np.array(variances))
    return max(slope, 0.1), max(intercept, 0.1)
```
