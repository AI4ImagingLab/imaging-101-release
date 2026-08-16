# Noise variance map

```python
import numpy as np
from scipy.ndimage import uniform_filter

def noise_variance(x, beta1, beta2, bg=100.0, filter_size=5):
    x = x.astype(np.float64)
    signal = uniform_filter(x - bg, size=filter_size)
    signal = np.maximum(signal, 0.0)
    return beta1 * signal + beta2
```

Notes:
- `uniform_filter` default mode is 'reflect'; keep the default.
- The clamp `max(signal, 0)` is applied to the background-subtracted local mean, not to the raw image.
