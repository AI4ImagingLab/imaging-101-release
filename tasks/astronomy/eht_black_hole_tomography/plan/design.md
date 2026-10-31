# Design: EHT Black Hole Tomography (BH-NeRF)

## Module Overview

```
data/raw_data.npz ──→ preprocessing.py ──→ obs_data dict
                                              │
data/meta_data ──→ preprocessing.py ──→ metadata dict
                                              │
                                         solvers.py
                                      BHNeRFSolver.reconstruct()
                                              │
                           ┌──────────────────┼──────────────────┐
                           │                  │                  │
                    physics_model.py   solvers.py (MLP)   solvers.py (loss)
                    ForwardModel       BHNeRFModel         loss_fn_image
                           │                  │                  │
                           └──────────────────┼──────────────────┘
                                              │
                                    visualization.py
                                    compute_metrics()
```

## Function Signatures

### src/physics_model.py

```python
def rotation_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation. axis: (3,), angle: (*batch). Returns (3, 3, *batch)."""

def keplerian_omega(r: np.ndarray, spin: float = 0.0, M: float = 1.0) -> np.ndarray:
    """Keplerian angular velocity. Returns same shape as r."""

def velocity_warp_coords(coords: torch.Tensor, Omega: torch.Tensor,
                         t_frame: float, t_start_obs: float,
                         t_geo: torch.Tensor, t_injection: float,
                         rot_axis: torch.Tensor, GM_c3: float = 1.0
                         ) -> torch.Tensor:
    """Velocity warp. coords: (3, *spatial). Returns (*spatial, 3)."""

def fill_unsupervised(emission: torch.Tensor, coords: torch.Tensor,
                      rmin: float, rmax: float, z_width: float
                      ) -> torch.Tensor:
    """Zero emission outside valid region. Returns same shape as emission."""

def trilinear_interpolate(volume: torch.Tensor, coords: torch.Tensor,
                          fov_min: float, fov_max: float) -> torch.Tensor:
    """Interpolate 3D volume at coords. volume: (D,H,W), coords: (*,3). Returns (*)."""

def volume_render(emission: torch.Tensor, g: torch.Tensor,
                  dtau: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
    """Integrate emission along rays. All inputs: (*spatial, ngeo). Returns (*spatial)."""

def dft_matrix(uv_coords: np.ndarray, fov_rad: float, npix: int) -> np.ndarray:
    """Build DFT matrix A. Returns (n_baselines, npix*npix) complex."""
```

### src/generate_data.py

```python
def schwarzschild_ray_paths(inclination: float, fov_M: float,
                            num_alpha: int, num_beta: int, ngeo: int
                            ) -> dict:
    """Compute ray paths. Returns dict with x,y,z,r,t_geo,dtau,Sigma arrays."""

def generate_gaussian_hotspot(resolution: int, rot_axis: np.ndarray,
                              rot_angle: float, orbit_radius: float,
                              std: float, fov_M: float) -> np.ndarray:
    """Generate 3D Gaussian hotspot. Returns (resolution, resolution, resolution)."""

def compute_doppler_factor(r: np.ndarray, theta: np.ndarray,
                           Omega: np.ndarray) -> np.ndarray:
    """Simplified Doppler factor for Schwarzschild. Returns same shape."""

def generate_dataset(meta_data_path: str, output_path: str) -> None:
    """Generate full synthetic dataset and save to npz."""
```

### src/preprocessing.py

```python
def load_metadata(data_dir: str = "data") -> dict:
def load_observation(data_dir: str = "data") -> dict:
def load_ground_truth(data_dir: str = "data") -> dict:
def prepare_data(data_dir: str = "data") -> tuple:
    """Returns (obs_data, ground_truth, metadata)."""
```

### src/solvers.py

```python
def positional_encoding(x: torch.Tensor, deg: int = 3) -> torch.Tensor:
    """Concatenate x with sin/cos encoding. x: (*,D). Returns (*,D+2*D*deg)."""

class MLP(torch.nn.Module):
    def __init__(self, in_features, net_depth=4, net_width=128, out_channel=1, do_skip=True):
    def forward(self, x: torch.Tensor) -> torch.Tensor:

class BHNeRFModel(torch.nn.Module):
    def __init__(self, scale, rmin, rmax, z_width, posenc_deg=3, net_depth=4, net_width=128):
    def forward(self, t_frame, coords, Omega, t_start_obs, t_geo, t_injection, rot_axis) -> torch.Tensor:

def loss_fn_image(pred: torch.Tensor, target: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
def loss_fn_lightcurve(pred: torch.Tensor, target: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
def loss_fn_visibility(pred_vis: torch.Tensor, target_vis: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:

class BHNeRFSolver:
    def __init__(self, metadata: dict):
    def reconstruct(self, obs_data: dict) -> dict:
    def predict_emission_3d(self, fov_M: float, resolution: int = 64) -> np.ndarray:
    def predict_movie(self, obs_data: dict) -> np.ndarray:
```

### src/visualization.py

```python
def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict:
def compute_image_metrics(pred_movie: np.ndarray, true_movie: np.ndarray) -> dict:
def plot_emission_slices(emission_3d, fov_M, ground_truth=None, save_path=None):
def plot_movie_comparison(pred_movie, true_movie, t_frames, n_show=6, save_path=None):
def plot_lightcurve(pred_images, true_images, t_frames, save_path=None):
def plot_loss_curves(loss_history, save_path=None):
def print_metrics_table(metrics: dict):
```
