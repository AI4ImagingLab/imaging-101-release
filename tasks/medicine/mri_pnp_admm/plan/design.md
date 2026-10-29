# Design: PnP-ADMM CS-MRI Reconstruction

## Module Architecture

```
main.py
  ├── src/preprocessing.py   → Load image, masks, noise from npz
  ├── src/physics_model.py   → CS-MRI forward model, data fidelity proximal
  ├── src/denoiser.py        → RealSN-DnCNN architecture and loading
  ├── src/solvers.py         → PnP-ADMM reconstruction loop
  └── src/visualization.py   → PSNR, NCC, NRMSE metrics + plots
```

## Data Flow

```
raw_data.npz + ground_truth.npz
        │
        ▼
preprocessing.prepare_data()
  ├── im_orig (256, 256)
  ├── mask (256, 256)
  └── noises (256, 256) complex
        │
        ▼
denoiser.load_denoiser("RealSN_DnCNN_noise15.pth")
        │
        ▼
solvers.pnp_admm_reconstruct()
  ├── physics_model.simulate_observation() → y
  ├── Loop 100 iters:
  │   ├── physics_model.data_fidelity_proximal() → v
  │   ├── denoiser forward pass → x
  │   └── dual update → u
  └── Returns reconstruction, zerofill, psnr_history
        │
        ▼
visualization.compute_metrics() + plot_*()
```

## Function Signatures

### src/preprocessing.py
```python
def load_observation(data_dir) -> dict:
def load_ground_truth(data_dir) -> ndarray:
def load_metadata(data_dir) -> dict:
def get_complex_noise(obs_data, scale=3.0) -> ndarray:
def get_mask(obs_data, mask_name="random") -> ndarray:
def prepare_data(data_dir, mask_name="random") -> tuple[ndarray, ndarray, ndarray, dict]:
```

### src/physics_model.py
```python
def forward_model(image, mask) -> ndarray:
def add_noise(kspace, noises) -> ndarray:
def simulate_observation(image, mask, noises) -> ndarray:
def zero_filled_recon(y) -> ndarray:
def data_fidelity_proximal(vtilde, y, mask, alpha) -> ndarray:
```

### src/denoiser.py
```python
class RealSN_DnCNN(nn.Module):
    def __init__(channels=1, num_of_layers=17):
    def forward(x: Tensor) -> Tensor:

def load_denoiser(weights_path, device="cpu") -> nn.Module:
```

### src/solvers.py
```python
def pnp_admm_reconstruct(model, im_orig, mask, noises, alpha, sigma, maxitr, device) -> dict:
```

### src/visualization.py
```python
def compute_psnr(estimate, reference) -> float:
def compute_metrics(estimate, reference) -> dict:
def plot_reconstruction_comparison(recon, zerofill, ground_truth, save_path):
def plot_error_maps(recon, zerofill, ground_truth, save_path):
def plot_psnr_convergence(psnr_history, save_path):
def plot_mask(mask, title, save_path):
def print_metrics(metrics, label):
```
