# Recorruption

```python
import numpy as np
from scipy.ndimage import uniform_filter

def recorrupt(y, beta1, beta2, alpha=1.5, bg=100.0, filter_size=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    y = y.astype(np.float64)
    signal = np.maximum(uniform_filter(y - bg, size=filter_size), 0.0)
    sigma2 = beta1 * signal + beta2
    g = rng.standard_normal(y.shape) * np.sqrt(sigma2)
    y_hat = y + alpha * g
    y_bar = y - g / alpha
    return y_hat, y_bar
```

Key constant: alpha = 1.5.
