Weighted least squares (numpy).

```python
import numpy as np

def linear_solve(A, data_2d, background_rms, exp_time):
    d = image2array(data_2d)
    C_D = background_rms**2 + np.abs(d) / exp_time
    w = 1.0 / C_D
    Aw = A * w[None, :]
    M = Aw.dot(A.T)
    R = Aw.dot(d)
    cond = np.linalg.cond(M)
    if cond < 1e14:
        params = np.linalg.solve(M, R)
    else:
        params, _, _, _ = np.linalg.lstsq(M, R, rcond=None)
    model_1d = A.T.dot(params)
    return params, array2image(model_1d)

param_lens, model_lens = linear_solve(A_lens, image_real, 10.0, 100.0)
```

This is the closed-form normal-equations solution c_hat = (A W A^T)^-1 A W d with diagonal W_ii = 1/C_D_i.