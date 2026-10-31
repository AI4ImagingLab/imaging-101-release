# Design: Code Architecture

## Module Overview

```
src/
├── physics_model.py    # Forward model: A v = crop(h * v)
├── preprocessing.py    # Data loading and normalisation
├── solvers.py          # ADMM solver and helpers
└── visualization.py    # Plotting and metrics
```

---

## physics_model.py

### `class RealFFTConvolve2D`

Implements the lensless forward operator and its adjoint using rfft2.

```python
class RealFFTConvolve2D:
    def __init__(self, psf: np.ndarray, norm: str = "ortho") -> None:
        """
        psf : (H, W, C) float32, normalised PSF.
        Precomputes:
          self._H       : rfft2(pad(psf))       # (pH, pW//2+1, C)
          self._Hadj    : conj(self._H)
          self._padded_shape : (pH, pW)
          self._start, self._end : crop indices
        """

    def convolve(self, v: np.ndarray) -> np.ndarray:
        """A v = crop( ifft(H · fft(pad(v))) )  →  (H, W, C)"""

    def deconvolve(self, y: np.ndarray) -> np.ndarray:
        """A^H y = crop( ifft(H* · fft(pad(y))) )  →  (H, W, C)"""

    def _pad(self, v: np.ndarray) -> np.ndarray:
        """Zero-pad (H,W,C) → (pH,pW,C)."""

    def _crop(self, x: np.ndarray) -> np.ndarray:
        """Crop (pH,pW,C) → (H,W,C), extracting start:end slice."""
```

---

## preprocessing.py

### `load_image(path, downsample) -> np.ndarray`
Load PNG → float32 [0,1], optional integer downsampling.

### `preprocess_psf(psf) -> np.ndarray`
Subtract min, divide by max. Returns float32 in [0, 1].

### `preprocess_measurement(data, psf) -> np.ndarray`
Subtract PSF min (dark level), divide by PSF max.

### `load_data(psf_path, data_path, downsample) -> tuple[ndarray, ndarray]`
High-level loader: reads images, applies downsample + normalisation.
Returns `(psf, measurement)` each of shape `(H, W, 3)`.

### `load_npz(npz_path) -> tuple[ndarray, ndarray]`
Load from `raw_data.npz`: reads `psf` and `measurement` keys,
removes batch dim, returns `(H, W, C)` arrays.

---

## solvers.py

### `soft_thresh(x, thresh) -> np.ndarray`
Element-wise: `sign(x) * max(|x| - thresh, 0)`.

### `finite_diff(v) -> np.ndarray`
Input `(H, W, C)` → output `(2, H, W, C)`.
`[v - roll(v,1,axis=0), v - roll(v,1,axis=1)]`

### `finite_diff_adj(u) -> np.ndarray`
Input `(2, H, W, C)` → output `(H, W, C)`.
`(u[0] - roll(u[0],-1,ax=0)) + (u[1] - roll(u[1],-1,ax=1))`

### `finite_diff_gram(shape) -> np.ndarray`
Input `(H, W)` → output `(H, W//2+1, 1)` complex.
`rfft2` of Laplacian kernel: `gram[0,0]=4, gram[0,1]=gram[0,-1]=gram[1,0]=gram[-1,0]=-1`.

### `class ADMM`

```python
class ADMM:
    def __init__(self, psf, mu1=1e-6, mu2=1e-5, mu3=4e-5, tau=2e-5) -> None:
        """Creates RealFFTConvolve2D internally."""

    def set_data(self, measurement: np.ndarray) -> None:
        """
        Sets measurement, precomputes:
          self._X_divmat    : 1/(crop_mask + mu1)  shape (pH, pW, C)
          self._Cty_padded  : zero-padded b         shape (pH, pW, C)
          self._R_divmat    : 1/(mu1|H|²+mu2|Ψ̃|²+mu3)  shape (pH,pW//2+1,C)
        Then calls reset().
        """

    def reset(self) -> None:
        """Initialise v=0.5, u=x=w=ξ=η=ρ=0."""

    def _U_update(self) -> None:  # u = soft_thresh(Ψv + η/μ₂, τ/μ₂)
    def _X_update(self) -> None:  # x = X_divmat * (ξ + μ₁Mv + Cᵀb)
    def _W_update(self) -> None:  # w = max(ρ/μ₃ + v, 0)
    def _image_update(self) -> None:  # v = ifft(R_divmat * fft(r))
    def _xi_update(self) -> None:     # ξ += μ₁(Mv - x)
    def _eta_update(self) -> None:    # η += μ₂(Ψv - u)
    def _rho_update(self) -> None:    # ρ += μ₃(v - w)
    def _update(self) -> None:        # one full iteration

    def apply(self, n_iter=100, verbose=True) -> np.ndarray:
        """Run n_iter ADMM iterations, return clip(v, 0, None)."""

    def get_image(self) -> np.ndarray:
        """Return current estimate clipped to [0, ∞)."""
```

---

## visualization.py

### `normalise_for_display(img) -> np.ndarray`
Shift to zero, divide by max. Returns [0, 1].

### `gamma_correction(img, gamma=2.2) -> np.ndarray`
Clip to [0,1] then raise to power 1/gamma.

### `plot_overview(psf, measurement, reconstruction, gamma, save_path) -> Figure`
Three-panel: PSF | raw measurement | reconstruction.

### `plot_convergence(residuals, save_path) -> Figure`
Log-scale plot of residual vs iteration.

### `compute_metrics(reconstruction, ground_truth) -> dict`
Returns `{"mse": …, "psnr": …, "ssim": …}`.
Uses `skimage.metrics`.
