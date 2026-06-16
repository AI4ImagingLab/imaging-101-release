# Design: MRI CG-SENSE Reconstruction

## Module Architecture
```
main.py
  ├── src/preprocessing.py    → Load k-space, sens maps, undersample
  ├── src/physics_model.py    → SENSE forward/adjoint, centered FFT/IFFT
  ├── src/solvers.py          → CG-SENSE (ported conjugate gradient solver)
  └── src/visualization.py    → Metrics + plots
```

## Function Signatures

### src/solvers.py
```python
def conjugate_gradient(matvec, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None) -> (x, info):
    """CG solver for Hermitian positive-definite systems (ported from scipy)."""

def cgsense_reconstruct(kspace_us, sens, coil_axis=-1) -> ndarray:  # complex image
def cgsense_image_recon(kspace_us, sens) -> ndarray:  # normalized magnitude
```

### src/physics_model.py
```python
def sense_forward(x, sens, mask) -> ndarray:  # image → multi-coil k-space
def sense_adjoint(y, sens) -> ndarray:        # k-space → image
def zero_filled_recon(kspace_us) -> ndarray:  # RSS baseline
```
