Grid and array conventions (numpy only).

```python
import numpy as np

def make_grid(numPix, deltapix):
    N = int(numPix)
    a = np.arange(N)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    x = (matrix[:, 0] - (N - 1) / 2.0) * deltapix
    y = (matrix[:, 1] - (N - 1) / 2.0) * deltapix
    return x, y  # each length N*N

def image2array(image):
    return image.reshape(image.shape[0] * image.shape[1])

def array2image(array):
    n = int(np.sqrt(len(array)))
    return array.reshape(n, n)

def re_size(image, factor):
    if factor == 1:
        return image.copy()
    f = int(factor)
    nx, ny = image.shape
    return image.reshape(nx // f, f, ny // f, f).mean(3).mean(1)
```

Notes:
- np.meshgrid default indexing is 'xy'; x varies with the column index and y with the row index. This exact pairing must match array2image (row-major).
- Grids used: detector 64x64 at 0.05; supersampled 320x320 at 0.01 (= 0.05/5). Block factor 5 divides 320 -> 64.