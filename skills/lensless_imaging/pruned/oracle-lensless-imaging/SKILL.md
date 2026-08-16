---
name: oracle-lensless-imaging
description: "Reconstruct a scene image from a single lensless (DiffuserCam) sensor measurement by inverting a diffuser-encoded cropped convolution, using an FFT-based ADMM solver with total-variation and non-negativity priors."
---

# oracle-lensless-imaging Oracle Skill

This skill reconstructs an RGB scene image v from a lensless sensor measurement b and a calibration point-spread function (PSF) h. The forward model is a shift-invariant linear system b = crop(h * v), where * is 2-D linear convolution and crop trims the (2H-1)x(2W-1) result to the central sensor extent HxW. The reconstruction solves the regularised non-negative least-squares problem min_{v>=0} (1/2)||b - Av||^2 + tau*||grad v||_1 with an anisotropic total-variation penalty, via the Alternating Direction Method of Multipliers (ADMM) with variable splitting. Each operation below is an independently removable piece of the pipeline: reading the public contract, loading the preprocessed arrays, establishing padding/crop geometry, building the FFT forward operator and its adjoint, defining the finite-difference gradient and its frequency-domain Gram spectrum, precomputing the diagonal solve factors, running the four proximal sub-updates plus dual ascent, and finally cropping/clipping and writing the output npz. Follow the operations in order; later operations assume the conventions established earlier.

## FFT forward convolution operator and adjoint

## FFT forward operator M and adjoint M^H

Build the shift-invariant convolution operator in the Fourier domain from the center-padded PSF. Use real FFTs (`rfft2`/`irfft2`) over the two spatial axes with `norm="backward"` (no scaling on forward transform, 1/N on inverse), which matches the standard convolution theorem and the LenslessPiCam default.

**Precompute once** from the PSF:
```python
from scipy.fft import rfft2, irfft2, ifftshift
norm = "backward"
psf_pad = pad(psf)                                   # (pH, pW, C)
H_fft   = rfft2(psf_pad, axes=(0, 1), norm=norm)     # (pH, pW//2+1, C) complex
Hadj    = np.conj(H_fft)
```

**Forward** `M x` (operand is already padded, output stays padded):
```python
def convolve(x_padded):
    Fx = rfft2(x_padded, axes=(0, 1), norm=norm)
    out = irfft2(H_fft * Fx, axes=(0, 1), s=tuple(padded_shape), norm=norm)
    return ifftshift(out, axes=(0, 1))
```

**Adjoint** `M^H y` (uses the conjugate spectrum):
```python
def deconvolve(y_padded):
    Fy = rfft2(y_padded, axes=(0, 1), norm=norm)
    out = irfft2(Hadj * Fy, axes=(0, 1), s=tuple(padded_shape), norm=norm)
    return ifftshift(out, axes=(0, 1))
```

Critical conventions:
- The `ifftshift` on the spatial axes after `irfft2` is REQUIRED; it corrects the phase offset introduced by center-padding the PSF. Omitting it misaligns the reconstruction.
- FFTs are taken over axes `(0, 1)` only; the channel axis (2) is untouched (per-channel independent convolution).
- `convolve`/`deconvolve` expect padded `(pH, pW, C)` inputs and return padded outputs; they do NOT add padding or crop.
- A standalone `forward(v)` on an unpadded image is `crop(convolve(pad(v)))`; use it for reprojection/diagnostics only, not inside the ADMM loop.

## Crop, clip, and write reconstruction npz

## Finalise and write output

After the ADMM loop, convert the padded estimate into the required output image:

```python
img = crop(image_est)          # (pH, pW, C) -> (760, 1014, 3)
img = np.clip(img, 0, None)    # enforce non-negativity
reconstruction = img.astype(np.float32)
```

Then save one npz per case to the run-local outputs directory:
```python
import os, numpy as np
os.makedirs("outputs", exist_ok=True)
np.savez(f"outputs/{case_id}.npz", reconstruction=reconstruction)
```

Requirements to satisfy the schema:
- key must be exactly `reconstruction`;
- shape exactly `(760, 1014, 3)` (this is what `crop` produces from the padded estimate);
- dtype `float32` (also acceptable `float64`); all values finite and non-negative;
- do NOT apply display-only transforms (gamma correction, min-max display normalisation) to the saved array — those are for visualisation only. Save the raw cropped/clipped reconstruction.

Process every case in the manifest, writing one `outputs/{case_id}.npz` each.
