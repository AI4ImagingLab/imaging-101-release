Noise model and observed image (numpy).

```python
import numpy as np

def add_poisson_noise(image, exp_time):
    sigma = np.sqrt(np.abs(image) / exp_time)
    return np.random.randn(*image.shape) * sigma

def add_background_noise(image, sigma_bkg):
    return np.random.randn(*image.shape) * sigma_bkg

np.random.seed(42)
image_real = (image_no_noise
              + add_poisson_noise(image_no_noise, 100.0)
              + add_background_noise(image_no_noise, 10.0))
```

The same variance model `C_D = background_rms**2 + |value|/exp_time` defines both the inversion weights (evaluated on the data d) and the reduced residuals (evaluated on the model).