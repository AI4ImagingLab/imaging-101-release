# Design: FPM-INR Code Architecture

## Module Organization

```
src/
├── preprocessing.py    # Data loading, optical parameter computation
├── physics_model.py    # FPM forward model (sub-spectrum extraction)
├── network.py          # INR architecture (FullModel, G_Tensor3D)
├── solvers.py          # Training loop + inference
├── visualization.py    # Plotting + metrics computation
└── all_in_focus.py     # Normal Variance focus stacking
```

## Data Flow

```
raw .mat files + meta_data JSON
    │
    ▼
preprocessing.py: load_raw_data() → compute_optical_params() → compute_pupil_and_propagation()
    │
    ▼
physics_model.py: FPMForwardModel(Pupil0, kzz, ledpos_true, M, N, MAGimg)
    │
    ▼
network.py: FullModel(w, h, num_feats, x_mode, y_mode, z_min, z_max, ...)
    │
    ▼
solvers.py: FPMINRSolver.train(model, forward_model, Isum, z_params)
    │
    ▼
solvers.py: FPMINRSolver.evaluate(model, z_positions) → (amplitude, phase)
    │
    ├─→ visualization.py: compute_metrics(pred, gt) → per-slice L2/PSNR/SSIM
    │
    └─→ all_in_focus.py: all_in_focus_normal_variance(z_stack) → AIF image
         │
         └─→ visualization.py: compute_allfocus_l2(aif_pred, aif_gt) → MSE
```

## Function Signatures

### preprocessing.py

```python
def load_raw_data(data_dir: str = "data") -> dict:
    # Returns: {'I_low', 'na_calib', 'mag', 'dpix_c', 'na_cal'}

def load_ground_truth(data_dir: str = "data") -> dict:
    # Returns: {'I_stack': (H, W, n_z), 'zvec': (n_z,)}

def load_metadata(data_dir: str = "data") -> dict:
    # Returns: full metadata dict from JSON

def compute_optical_params(raw_data: dict, metadata: dict) -> dict:
    # Returns: {'Fxx1', 'Fyy1', 'ledpos_true', 'Isum', 'order', 'u', 'v',
    #           'M', 'N', 'MM', 'NN', 'k0', 'kmax', 'D_pixel', 'MAGimg', 'ID_len'}

def compute_pupil_and_propagation(optical_params: dict) -> dict:
    # Returns: {'Pupil0': (M, N), 'kzz': (M, N) complex}

def compute_z_params(metadata: dict, optical_params: dict) -> dict:
    # Returns: {'DOF', 'delta_z', 'num_z', 'z_min', 'z_max'}

def prepare_data(data_dir: str, device: str) -> dict:
    # Returns: {'metadata', 'optical_params', 'pupil_data', 'z_params', 'Isum'}
```

### physics_model.py

```python
class FPMForwardModel:
    def __init__(self, Pupil0, kzz, ledpos_true, M, N, MAGimg): ...
    def compute_spectrum_mask(self, dz: Tensor, led_num: list) -> Tensor: ...
    def get_led_coords(self, led_num: list) -> tuple: ...
    def get_sub_spectrum(self, img_complex: Tensor, led_num: list, spectrum_mask: Tensor) -> Tensor: ...
    def get_measured_amplitudes(self, Isum: Tensor, led_num: list, n_z: int) -> Tensor: ...
```

### network.py

```python
class G_Renderer(nn.Module):       # MLP: features → scalar
class G_FeatureTensor(nn.Module):  # 2D learnable feature grid
class G_Tensor(G_FeatureTensor):   # 2D features + renderer
class G_Tensor3D(nn.Module):       # 3D factored representation
class FullModel(nn.Module):        # Dual G_Tensor3D for amplitude + phase
    def forward(self, dz: Tensor) -> tuple[Tensor, Tensor]: ...

def save_model_with_required_grad(model, save_path): ...
def load_model_with_required_grad(model, load_path): ...
```

### solvers.py

```python
class FPMINRSolver:
    def __init__(self, num_epochs, lr, lr_decay_step, lr_decay_gamma, use_amp, use_compile): ...
    def train(self, model, forward_model, Isum, z_params, device, vis_callback) -> dict: ...
    def evaluate(self, model, z_positions, device, chunk_size) -> tuple[ndarray, ndarray]: ...
```

### visualization.py

```python
def compute_metrics(pred_stack, gt_stack) -> dict: ...
def compute_ssim_per_slice(pred_norm, gt_norm) -> ndarray: ...
def compute_allfocus_l2(aif_pred, aif_gt) -> dict: ...
def plot_amplitude_phase(amplitude, phase, epoch, save_path, figsize) -> Figure: ...
def plot_per_slice_metrics(z_positions, l2, psnr, ssim, save_path) -> Figure: ...
def plot_gt_comparison(pred_norm, gt_norm, z_positions, psnr, l2, save_path) -> Figure: ...
def plot_allfocus_comparison(aif_pred, aif_gt, l2_error, save_path) -> Figure: ...
```

### all_in_focus.py

```python
def create_balance_map(image_size, patch_size, patch_pace) -> tuple[int, ndarray]: ...
def all_in_focus_normal_variance(z_stack, patch_size, patch_pace) -> ndarray: ...
```
