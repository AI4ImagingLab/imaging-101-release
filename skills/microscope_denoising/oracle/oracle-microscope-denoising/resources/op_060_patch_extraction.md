# Patch extraction

```python
import numpy as np

def extract_patches(images, recorrupt, patch_size=128, n_patches=10000,
                    beta1=1.0, beta2=0.5, alpha=1.5, bg=100.0, seed=None):
    rng = np.random.default_rng(seed)
    N, H, W = images.shape
    per = max(1, n_patches // N)
    yh, yb = [], []
    for img in images.astype(np.float64):
        for _ in range(per):
            i = rng.integers(0, H - patch_size)
            j = rng.integers(0, W - patch_size)
            p = img[i:i+patch_size, j:j+patch_size]
            p = np.rot90(p, rng.integers(4))
            if rng.random() < 0.5:
                p = np.fliplr(p)
            y_hat, y_bar = recorrupt(p, beta1, beta2, alpha, bg, rng=rng)
            lo = np.percentile(y_hat, 0); hi = np.percentile(y_hat, 100)
            span = hi - lo if hi > lo else 1e-6
            yh.append(np.clip((y_hat - lo)/span, 0, 1)[np.newaxis].astype(np.float32))
            yb.append(np.clip((y_bar - lo)/span, 0, 1)[np.newaxis].astype(np.float32))
    return np.stack(yh), np.stack(yb)   # (N,1,128,128) each
```

Constants: patch_size=128, n_patches~=10000, eps=1e-6.
