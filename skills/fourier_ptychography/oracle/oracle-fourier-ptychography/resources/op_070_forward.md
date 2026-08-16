# Forward model reference

```python
import numpy as np
# fft2c / ifft2c defined per the FFT-convention operation

def forward_single(object_spectrum, pupil, pos_j, Np):
    r0, c0 = int(pos_j[0]), int(pos_j[1])
    patch = object_spectrum[r0:r0+Np, c0:c0+Np]
    esw = patch * pupil
    ESW = fft2c(esw)
    Iest = np.abs(ESW)**2
    return patch, esw, ESW, Iest
```

Alternative equivalent simulation (spectrum built once via fft2c of the object,
sub-patch multiplied by pupil, then plain centered ifft to image plane) yields
the same intensities; keep whichever pair you use consistent between forward and
back-propagation. The measured stack has one such intensity per LED.
