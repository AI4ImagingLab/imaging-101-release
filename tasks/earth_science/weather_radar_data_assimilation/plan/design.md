# Design: Weather Radar Data Assimilation

## File Structure

```
src/
├── preprocessing.py      # Data loading and normalization
├── physics_model.py      # Forward measurement model (masking + noise)
├── solvers.py            # Flow-based sampler with guided EM
├── visualization.py      # Weather radar visualization with VIL colormap
└── generate_data.py      # Synthetic observation generation
```

## Function Signatures

### preprocessing.py

```python
def load_raw_data(data_dir: str) -> dict:
    """Load raw_data.npz and return dict with condition_frames, observations, observation_mask."""

def load_ground_truth(data_dir: str) -> np.ndarray:
    """Load ground_truth.npz and return target_frames array."""

def load_meta_data(data_dir: str) -> dict:
    """Load meta_data.json."""

def scale_to_model(x: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0,1] to model space: (x - 0.5) * 10."""

def scale_from_model(x: np.ndarray) -> np.ndarray:
    """Scale from model space back to [0,1]: x / 10 + 0.5."""
```

### physics_model.py

```python
def make_observation_operator(mask: np.ndarray) -> callable:
    """Return a function that applies the binary mask to an input array."""

def make_noiser(sigma: float) -> callable:
    """Return a function that adds Gaussian noise with given sigma."""

def forward_model(x: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Apply full forward model: y = mask * x + N(0, sigma^2)."""
```

### solvers.py

```python
class StochasticInterpolant:
    """Defines interpolant coefficients alpha(t), beta(t), sigma(t) for the flow."""
    def __init__(self, beta_fn: str = "t^2", sigma_coef: float = 1.0): ...
    def interpolant_coefs(self, D: dict) -> dict: ...
    def sigma(self, t: torch.Tensor) -> torch.Tensor: ...

class DriftModel(torch.nn.Module):
    """UNet-based drift model for the stochastic interpolant."""
    def __init__(self, config): ...
    def forward(self, zt, t, y, cond=None) -> torch.Tensor: ...

def guided_em_sample(
    model: DriftModel,
    interpolant: StochasticInterpolant,
    base: torch.Tensor,
    cond: torch.Tensor,
    observation: torch.Tensor,
    operator: callable,
    noiser: callable,
    n_steps: int = 500,
    mc_times: int = 25,
    guidance_scale: float = 0.1,
) -> torch.Tensor:
    """Run guided Euler-Maruyama sampling to reconstruct a single frame."""

def autoregressive_reconstruct(
    model: DriftModel,
    interpolant: StochasticInterpolant,
    condition_frames: torch.Tensor,
    observations: torch.Tensor,
    operator: callable,
    noiser: callable,
    n_steps: int = 500,
    mc_times: int = 25,
    auto_steps: int = 3,
) -> torch.Tensor:
    """Autoregressively reconstruct multiple future frames."""
```

### visualization.py

```python
def plot_comparison(
    condition: np.ndarray,
    ground_truth: np.ndarray,
    observations: np.ndarray,
    reconstruction: np.ndarray,
    save_path: str,
) -> None:
    """Plot side-by-side comparison of condition, GT, observations, and reconstruction."""

def get_vil_colormap() -> matplotlib.colors.ListedColormap:
    """Return the standard VIL radar colormap."""
```

### generate_data.py

```python
def generate_observations(
    ground_truth: np.ndarray,
    mask_ratio: float = 0.1,
    noise_sigma: float = 0.001,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sparse observations from ground truth.
    Returns (observations, mask)."""
```

## Data Flow

```
raw_data.npz ──→ preprocessing.py ──→ condition_frames (scaled)
                                   ──→ observations (scaled)
                                   ──→ observation_mask
                                          │
meta_data.json ──→ preprocessing.py ──→ config params
                                          │
          ┌───────────────────────────────┘
          │
physics_model.py ──→ observation_operator
                  ──→ noiser
                        │
          ┌─────────────┘
          │
solvers.py ──→ load pretrained UNet drift model
           ──→ guided_em_sample (per frame)
           ──→ autoregressive_reconstruct (3 frames)
                        │
                        ▼
              reconstructed_frames
                        │
          ┌─────────────┴────────────┐
          │                          │
visualization.py              evaluation/
  (comparison plots)          (NCC, NRMSE vs GT)
```
