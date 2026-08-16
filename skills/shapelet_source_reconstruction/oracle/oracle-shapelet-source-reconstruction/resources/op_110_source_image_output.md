Source reconstruction and output (numpy).

```python
import os
import numpy as np

numPix_hr = 64 * 5              # 320
deltaPix_hr = 0.05 / 5          # 0.01
x_hr, y_hr = make_grid(numPix_hr, deltaPix_hr)
source = shapelet_function(x_hr, y_hr, param_lens,
                           20, 0.15, 0.2, 0.0)
source_image = array2image(source) * deltaPix_hr**2   # (320, 320)

assert source_image.shape == (320, 320)
assert np.all(np.isfinite(source_image))
os.makedirs('outputs', exist_ok=True)
np.savez('outputs/%s.npz' % case_id,
         source_image=source_image.astype(np.float64))
```

The scale parameter (0.15) and center (0.2, 0.0) used here are identical to the response-matrix reconstruction configuration, not the n_max=150 simulation parameters.