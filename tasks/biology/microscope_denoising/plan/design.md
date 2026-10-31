# Design: Module Structure and Function Signatures

## Directory Layout

```
src/
├── physics_model.py   # Noise forward model
├── preprocessing.py   # Recorruption and patch extraction
├── solvers.py         # U-Net architecture and training loop
├── visualization.py   # Plotting and metrics
└── generate_data.py   # Data download and packaging
```

## `src/physics_model.py`

```python
def noise_variance(x, beta1, beta2, bg=100.0, filter_size=5) -> np.ndarray:
    """
    Per-pixel noise variance: sigma^2 = beta1 * max(H(x-bg), 0) + beta2.
    x: (..., H, W) float64; returns (..., H, W) float64.
    """

def add_noise(x, beta1, beta2, bg=100.0, filter_size=5, rng=None) -> np.ndarray:
    """Add Poisson-Gaussian noise to a clean image. Returns noisy y."""

def estimate_noise_params(y, bg=100.0, filter_size=5) -> tuple[float, float]:
    """
    Estimate (beta1, beta2) from single noisy image via linear regression of
    block-wise variance vs block-wise mean signal.
    """
```

## `src/preprocessing.py`

```python
def recorrupt(y, beta1, beta2, alpha=1.5, bg=100.0, filter_size=5,
              rng=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate (y_hat, y_bar) pair from a noisy image.
    y_hat = y + alpha*g, y_bar = y - g/alpha,  g ~ N(0, sigma^2).
    Both outputs have same shape as y (H, W), float64.
    """

def prctile_norm(x, pmin=0.0, pmax=100.0) -> np.ndarray:
    """Percentile normalisation to [0, 1]. Returns float32."""

def extract_patches(images, patch_size=128, n_patches=10000,
                    beta1=1.0, beta2=0.5, alpha=1.5, bg=100.0,
                    filter_size=5, seed=None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    images: (N, H, W) float64 stack.
    Returns y_hat_patches, y_bar_patches: both (N_patches, 1, patch_size, patch_size) float32.
    Applies random crop, rotation, flip, recorruption, and joint normalisation.
    """
```

## `src/solvers.py`

```python
class DoubleConv(nn.Module):
    """Two Conv2d(3×3)+ReLU blocks. __init__(in_ch, out_ch)."""

class UNet(nn.Module):
    """
    4-level encoder–decoder with skip connections.
    __init__(base=32): channels 1→32→64→128→256 (enc), 256→128→64→32→1 (dec).
    forward(x) -> x:  x shape (B, 1, H, W).
    """

def train_denoiser(y_hat, y_bar, n_iters=30000, batch_size=4,
                   lr=5e-4, lr_decay_steps=10000, lr_decay_factor=0.5,
                   base=32, device=None, verbose=True
                   ) -> tuple[UNet, list[float]]:
    """
    Train U-Net with MSE(f(y_hat), y_bar). Returns (trained_model_cpu, loss_history).
    loss_history: list of float, one value per 100 iterations.
    """

def denoise_image(model, y, patch_size=128, overlap=32, batch_size=8,
                  device=None) -> np.ndarray:
    """
    Patch-based inference on full image with linear-taper blending.
    y: (H, W) float64 raw image.
    Returns denoised: (H, W) float32, same intensity range as input.
    """
```

## `src/visualization.py`

```python
def compute_psnr(pred, gt, data_range=None) -> float:
def compute_ssim(pred, gt, data_range=None) -> float:
def compute_nrmse(pred, gt) -> float:
def compute_snr_improvement(noisy, denoised, gt) -> tuple[float, float, float]:
    """Returns (psnr_noisy, psnr_denoised, delta_psnr) in dB."""
def compute_all_metrics(pred, gt) -> dict:
    """Returns {'psnr': float, 'ssim': float, 'nrmse': float}."""

def plot_comparison(noisy, denoised, gt=None, titles=None,
                    pmin=1, pmax=99.5, figsize=None, save_path=None) -> plt.Figure:
def plot_zoom(noisy, denoised, gt=None, roi=None,
              pmin=1, pmax=99.5, figsize=None, save_path=None) -> plt.Figure:
def plot_training_curve(loss_history, log_scale=True, save_path=None) -> plt.Figure:
def plot_intensity_profile(noisy, denoised, gt=None, row=None,
                           save_path=None) -> plt.Figure:
```

## `main.py`

Top-level pipeline (sequential):
1. Load `data/raw_data.npz` and `data/meta_data.json`.
2. Call `estimate_noise_params` on `measurements[0]`.
3. Call `extract_patches` on all 10 frames.
4. Call `train_denoiser` → save model weights and loss history.
5. Call `denoise_image` on `measurements[0]` (test frame).
6. Load `data/ground_truth.npz`; call `compute_all_metrics`.
7. Call `plot_comparison`, `plot_zoom`, `plot_training_curve`.
8. Save `evaluation/reference_outputs/{denoised.npy, metrics.json, loss_history.npy, model_weights.pt}`.
