# Code Architecture: DPI Task

## Module Overview

```
src/
├── preprocessing.py    # Data I/O and closure index extraction
├── physics_model.py    # GPU NUFFT forward model + loss functions
├── solvers.py          # Real-NVP architecture + DPI training
├── visualization.py    # Posterior plots + quality metrics
└── generate_data.py    # Synthetic crescent generation
```

## Data Flow

```
obs.uvfits ──→ preprocessing ──→ closure_indices, nufft_params
                                          │
gt.fits    ──→ load_ground_truth          │
                                          ▼
meta_data  ──→ load_metadata ──→ DPISolver.reconstruct()
                                   │  ├─ Build NUFFTForwardModel
                                   │  ├─ Build RealNVP + Img_logscale
                                   │  └─ Training loop (30K epochs)
                                   ▼
                              DPISolver.sample()
                                   │  ├─ z ~ N(0,I) → flow.reverse() → Softplus
                                   │  └─ Return (n_samples, 32, 32)
                                   ▼
                              posterior_statistics()
                                   ├─ mean, std, samples
                                   ▼
                              visualization + metrics
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir="data") -> dict
    # Returns: {'obs': Obsdata, 'vis': (M,), 'vis_sigma': (M,), 'uv_coords': (M,2)}

def load_ground_truth(data_dir="data", npix=32, fov_uas=160.0) -> np.ndarray
    # Returns: (npix, npix) ground-truth image

def load_metadata(data_dir="data") -> dict
    # Returns: {npix, fov_uas, n_flow, n_epoch, batch_size, lr, ...}

def extract_closure_indices(obs, snrcut=0.0) -> dict
    # Returns: {cphase_ind_list, cphase_sign_list, camp_ind_list, cphase_data, ...}

def compute_nufft_params(obs, npix, fov_uas) -> dict
    # Returns: {ktraj_vis: (1,2,M) Tensor, pulsefac_vis: (2,M) Tensor}

def build_prior_image(obs, npix, fov_uas, prior_fwhm_uas=50.0) -> tuple
    # Returns: (prior_image: (npix,npix), flux_const: float)

def prepare_data(data_dir="data") -> tuple
    # Returns: (obs, obs_data, closure_indices, nufft_params, prior_image, flux_const, metadata)
```

### physics_model.py

```python
class NUFFTForwardModel(nn.Module):
    def __init__(self, npix, ktraj_vis, pulsefac_vis,
                 cphase_ind_list, cphase_sign_list, camp_ind_list, device)
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]
        # images: (B, npix, npix) → (vis, visamp, cphase, logcamp)

# Loss functions (all return closures):
def Loss_angle_diff(sigma, device) -> Callable     # Closure phase
def Loss_logca_diff2(sigma, device) -> Callable     # Log closure amp
def Loss_vis_diff(sigma, device) -> Callable        # Visibility
def Loss_logamp_diff(sigma, device) -> Callable     # Log amplitude
def Loss_l1(y_pred) -> Tensor                       # L1 sparsity
def Loss_TSV(y_pred) -> Tensor                      # Total squared variation
def Loss_flux(flux) -> Callable                     # Flux constraint
def Loss_center(device, center, dim) -> Callable    # Centering
def Loss_cross_entropy(y_true, y_pred) -> Tensor    # MEM
```

### solvers.py

```python
class ActNorm(nn.Module):     # Data-dependent normalization
class ZeroFC(nn.Module):      # Zero-init linear + learnable scale
class AffineCoupling(nn.Module):  # Split-transform coupling layer
class Flow(nn.Module):        # ActNorm → Coupling → Reverse (×2)
class RealNVP(nn.Module):     # Stack of Flows with permutations
class Img_logscale(nn.Module): # Learnable log-scale

class DPISolver:
    def __init__(self, npix=32, n_flow=16, ...)
    def reconstruct(self, obs_data, closure_indices, nufft_params,
                    prior_image, flux_const) -> dict
    def sample(self, n_samples=1000) -> np.ndarray  # (n, npix, npix)
    def posterior_statistics(self, n_samples=1000) -> dict  # mean, std, samples
```

### visualization.py

```python
def compute_metrics(estimate, ground_truth) -> dict
    # Returns: {nrmse, ncc, dynamic_range}

def compute_uq_metrics(posterior_mean, posterior_std, ground_truth) -> dict
    # Returns: {nrmse, ncc, dynamic_range, calibration, mean_uncertainty}

def plot_posterior_summary(mean, std, samples, ground_truth, ...)
def plot_posterior_samples(samples, n_show=8, ...)
def plot_loss_curves(loss_history, ...)
```
