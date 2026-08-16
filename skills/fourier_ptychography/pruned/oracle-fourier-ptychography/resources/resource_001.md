# k-space position mapping (reference)

```python
import numpy as np

def kspace_positions(encoder, z_led, wavelength, No, Np, dxo):
    """Return integer (J,2) upper-left corners of each LED's Np x Np pupil window
    inside the No x No centered spectrum (DC at center)."""
    conv = -(1.0 / wavelength) * dxo * Np
    dist = np.sqrt(encoder[:, 0]**2 + encoder[:, 1]**2 + z_led**2)
    shift = np.round(conv * encoder / dist[:, None])
    positions = (shift + No // 2 - Np // 2).astype(int)
    return positions

def derive_No(encoder, z_led, wavelength, Nd, dxo):
    conv = -(1.0 / wavelength) * dxo * Nd
    dist = np.sqrt(encoder[:, 0]**2 + encoder[:, 1]**2 + z_led**2)
    shifts = np.round(conv * encoder / dist[:, None]).astype(int)
    span = int(np.max(shifts.max(0) - shifts.min(0))) + 2 * Nd
    if span % 2 == 1:
        span += 1
    return max(Nd, span)
```

Notes:
- `dxo == dxp == dxd/magnification`.
- Ensure every `positions[j]` satisfies `0 <= pos <= No-Np` so patch slices stay in bounds; if not, `No` is too small or axis order is wrong.
