# Design: seismic_FWI_original

All three components (Ricker wavelet, cosine taper, acoustic wave solver) are
implemented from scratch without importing `deepwave`.

---

## src/physics_model.py

### FD stencils (internal)

```python
def _fd1_y(a: Tensor, rdy: float) -> Tensor:
    """4th-order first derivative along y (dim -2). Boundary → 0."""

def _fd1_x(a: Tensor, rdx: float) -> Tensor:
    """4th-order first derivative along x (dim -1). Boundary → 0."""

def _fd2_y(a: Tensor, rdy2: float) -> Tensor:
    """4th-order second derivative along y (dim -2). Boundary → 0."""

def _fd2_x(a: Tensor, rdx2: float) -> Tensor:
    """4th-order second derivative along x (dim -1). Boundary → 0."""
```

### CFL condition

```python
def cfl_step_ratio(dy, dx, dt, v_max, c_max=0.6) -> (inner_dt: float, step_ratio: int):
    """Compute step_ratio = ceil(dt / dt_max), dt_max = 0.6/sqrt(1/dy²+1/dx²)/v_max."""
```

### PML profile construction

```python
def _setup_pml_1d(n, pml_start_left, pml_start_right, pml_width,
                  sigma0, alpha0, inner_dt, dtype, device) -> (a: Tensor, b: Tensor):
    """1D C-PML profiles. Returns a, b of shape (n,). Interior: a=0, b=0."""

def setup_pml_profiles(ny_p, nx_p, pml_width, fd_pad, dy, dx, inner_dt,
                       v_max, dtype, device, pml_freq, r_val=0.001, n_power=2
                       ) -> List[Tensor]:
    """Returns [ay, by, dbydy, ax, bx, dbxdx].
    y-profiles shape (ny_p, 1), x-profiles shape (1, nx_p)."""
```

### Wave step

```python
def wave_step(v_p, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
              pml_profiles, dy, dx, inner_dt
              ) -> (new_wfc, new_psi_y, new_psi_x, new_zeta_y, new_zeta_x):
    """Single C-PML Verlet time step. All tensors: (ny_p, nx_p).
    Differentiable w.r.t. v_p via PyTorch autograd."""
```

### Resampling (internal)

```python
def _fft_upsample(signal: Tensor, step_ratio: int) -> Tensor:
    """FFT low-pass upsample of last dim. Matches deepwave.common.upsample."""

def _fft_downsample(signal: Tensor, step_ratio: int) -> Tensor:
    """FFT anti-aliased downsample of last dim. Matches deepwave.common.downsample."""
```

### Location helper (internal)

```python
def _loc_to_flat(loc, pad_y, pad_x, nx_p) -> LongTensor:
    """loc[..., 0]=row (slow/dim0), loc[..., 1]=col (fast/dim1).
    Flat index in padded model: (loc[0]+pad_y)*nx_p + (loc[1]+pad_x).
    Matches deepwave's row-major convention exactly."""
```

### Public API

```python
def make_acquisition_geometry(nx, n_shots=10, n_receivers=93,
                               source_depth=1, receiver_depth=1,
                               device=cpu) -> (source_loc, receiver_loc):
    """Uniform surface acquisition. Shapes: (n_shots,1,2), (n_shots,n_rec,2)."""

def make_ricker_wavelet(freq, nt, dt, n_shots, device=cpu) -> Tensor:
    """Ricker wavelet: (1-2π²f²(t-tp)²)*exp(-π²f²(t-tp)²). Shape (n_shots,1,nt)."""

def forward_model(v, spacing, dt, source_amp, source_loc, receiver_loc,
                  freq, accuracy=4, pml_width=20, checkpoint_every=64) -> Tensor:
    """Acoustic FWI forward operator. Processes shots sequentially.
    Returns receiver_data of shape (n_shots, n_rec, nt_user).
    v may have requires_grad=True for FWI gradient computation.

    Uses torch.utils.checkpoint every `checkpoint_every` inner steps to bound
    peak GPU memory (~250 MB per shot on Marmousi vs ~40 GB without checkpointing).
    When v.requires_grad=False, runs in torch.no_grad() for faster evaluation."""
```

---

## src/solvers.py

```python
def cosine_taper(x: Tensor, n_taper=5) -> Tensor:
    """Cosine taper on last n_taper samples: (cos(πi/n)+1)/2 for i=1..n."""

def smooth_gradient(grad: ndarray, sigma=1.0) -> ndarray:
    """Gaussian filter on velocity gradient (scipy.ndimage)."""

def run_fwi(v_init, spacing, dt, source_amp, source_loc, receiver_loc,
            observed_data, freq, n_epochs=800, lr=1e2, milestones=(75,300),
            v_min=1480, v_max=5800, device=cpu, print_every=50
            ) -> (v_inv: Tensor, losses: List[float]):
    """Adam FWI loop. Returns inverted velocity and loss history."""
```

---

## src/preprocessing.py, src/visualization.py

Identical to `seismic_FWI/src/` (no deepwave dependency in those files).

---

## main.py

CLI entry point. Arguments: `--epochs`, `--device`, `--output-dir`, `--data`.
Saves `v_inv.npy`, `losses.npy`, `pred_data.npy`, `metrics.json`, and figures.

---

## Key Differences from seismic_FWI

| Component | seismic_FWI | seismic_FWI_original |
|-----------|-------------|----------------------|
| Forward solver | `deepwave.scalar` (C/CUDA) | Pure PyTorch FD loop |
| Ricker wavelet | `deepwave.wavelets.ricker` | `make_ricker_wavelet` (inline formula) |
| Cosine taper | `deepwave.common.cosine_taper_end` | `cosine_taper` (inline formula) |
| Memory | O(1) per shot (C backend) | O(nt_inner × ny_p × nx_p) autograd graph |
| Shot batching | All shots simultaneously | Sequential (one shot at a time) |
| Speed | ~4 min / 800 epochs (GPU) | ~5-30× slower |
| Parity | reference | rel. L2 ≈ 1.67e-6 (float32 machine precision) |
