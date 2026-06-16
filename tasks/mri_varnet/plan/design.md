# Design: MRI VarNet Reconstruction

## Module Architecture

```
main.py
  ├── src/preprocessing.py    → Load k-space, apply undersampling mask
  ├── src/physics_model.py    → Center crop, zero-filled recon
  ├── src/solvers.py          → VarNet model loading and inference
  └── src/visualization.py    → SSIM, NCC, NRMSE metrics + plots
```

## Function Signatures

### src/preprocessing.py
```python
def load_observation(data_dir) -> dict:
def load_ground_truth(data_dir) -> ndarray:        # (N, 320, 320)
def get_complex_kspace(obs_data) -> ndarray:        # (N, Nc, H, W) complex64
def apply_mask(kspace_slice, acceleration, center_fraction, seed) -> (Tensor, Tensor):
def prepare_data(data_dir) -> (ndarray, ndarray, dict):
```

### src/solvers.py
```python
def load_varnet(weights_path, device) -> VarNet:
def varnet_reconstruct(model, masked_kspace, mask, device) -> ndarray:
def varnet_reconstruct_batch(model, kspace_slices, ...) -> (ndarray, ndarray):
```
