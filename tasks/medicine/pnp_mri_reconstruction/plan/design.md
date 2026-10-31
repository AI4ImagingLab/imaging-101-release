# Design: PnP-MSSN MRI Reconstruction

## File structure

```
tasks/pnp_mri_reconstruction/
├── README.md
├── requirements.txt
├── main.py                        # Pipeline entry point
├── data/
│   ├── raw_data.npz               # Ground truth MRI image (key: 'img')
│   └── meta_data                  # JSON: image_size, num_lines, patch_size, etc.
├── models/
│   └── checkpoints/               # Pre-trained MSSN weights (TF1 format)
├── plan/
│   ├── approach.md
│   └── design.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Data loading and preparation
│   ├── physics_model.py           # MRI forward model
│   ├── solvers.py                 # PnP-PGM solver + MSSN denoiser
│   └── visualization.py           # Plotting and metrics
├── evaluation/
│   ├── reference_outputs/
│   │   ├── ground_truth.npy
│   │   ├── ifft_recon.npy
│   │   ├── pnp_mssn_recon.npy
│   │   ├── snr_history.npy
│   │   ├── dist_history.npy
│   │   ├── sampling_mask.npy
│   │   └── metrics.json
│   ├── fixtures/
│   │   ├── preprocessing/
│   │   ├── physics_model/
│   │   └── solvers/
│   └── tests/
│       ├── test_preprocessing.py
│       ├── test_physics_model.py
│       ├── test_solvers.py
│       ├── test_visualization.py
│       └── test_end_to_end.py
├── notebooks/
│   └── pnp_mssn.ipynb
└── output/
    └── reconstruction.npy
```

## Module signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict:
    """Load MRI ground truth image.
    Returns: {'img': ndarray (N, N)}
    """

def load_metadata(data_dir: str = "data") -> dict:
    """Load imaging parameters.
    Returns: {'image_size': [N, N], 'num_lines': int, 'num_iter': int,
              'step_size': float, 'patch_size': int, 'stride': int,
              'state_num': int, 'sigma': int, 'model_checkpoints': str}
    """

def prepare_data(data_dir: str = "data") -> tuple:
    """Full data preparation pipeline.
    Returns: (img, mask, y, metadata)
        img: ndarray (N, N) — normalized ground truth
        mask: ndarray (N, N) — boolean sampling mask
        y: ndarray (N, N) — subsampled k-space measurements
        metadata: dict
    """
```

### physics_model.py

```python
class MRIForwardModel:
    def __init__(self, mask: ndarray):
        """Initialize with sampling mask.
        mask: (N, N) boolean array
        """

    def forward(self, x: ndarray) -> ndarray:
        """Forward model: image -> subsampled k-space.
        x: (N, N) image
        Returns: (N, N) complex k-space (zeros where mask is False)
        """

    def adjoint(self, z: ndarray) -> ndarray:
        """Adjoint: subsampled k-space -> image.
        z: (N, N) complex k-space
        Returns: (N, N) complex image
        """

    def grad(self, x: ndarray, y: ndarray) -> tuple:
        """Gradient of data fidelity 0.5*||Ax - y||^2.
        Returns: (gradient, cost)
        """

    @staticmethod
    def generate_mask(image_size: ndarray, num_lines: int) -> ndarray:
        """Generate radial sampling mask.
        Returns: (N, N) boolean mask
        """

    def ifft_recon(self, y: ndarray) -> ndarray:
        """Naive IFFT reconstruction from subsampled k-space.
        Returns: (N, N) real image
        """
```

### solvers.py

```python
class MSSNDenoiser:
    def __init__(self, image_shape: tuple, sigma: int = 5,
                 model_checkpoints: str = "models/checkpoints/mssn-550000iters",
                 patch_size: int = 42, stride: int = 7, state_num: int = 8):
        """Initialize MSSN denoiser with pre-trained weights."""

    def denoise(self, image: ndarray) -> ndarray:
        """Denoise a single grayscale image.
        image: (N, N) float in [0, 1]
        Returns: (N, N) denoised image in [0, 1]
        """

def pnp_pgm(forward_model: MRIForwardModel,
             denoiser: MSSNDenoiser,
             y: ndarray,
             num_iter: int = 200,
             step: float = 1.0,
             xtrue: ndarray = None,
             verbose: bool = True,
             save_dir: str = None) -> tuple:
    """Plug-and-Play Proximal Gradient Method.
    Returns: (reconstruction, history)
        reconstruction: (N, N) final image
        history: dict with keys 'snr', 'dist', 'time', 'relative_change'
    """
```

### visualization.py

```python
def compute_snr(ground_truth: ndarray, estimate: ndarray) -> float:
    """Compute SNR in dB."""

def compute_metrics(ground_truth: ndarray, estimate: ndarray) -> dict:
    """Compute reconstruction quality metrics.
    Returns: {'snr_db': float, 'nrmse': float, 'ncc': float}
    """

def plot_comparison(ground_truth, ifft_recon, pnp_recon, metrics, save_path=None):
    """Side-by-side comparison of reconstructions."""

def plot_convergence(history, save_path=None):
    """SNR and residual vs iteration."""

def plot_error_maps(ground_truth, ifft_recon, pnp_recon, mask, save_path=None):
    """Error maps and sampling pattern."""

def plot_progression(recon_dir, ground_truth, iters, save_path=None):
    """Reconstruction at selected iterations."""
```

## Data flow

```
raw_data.npz + meta_data
        |
        v
  preprocessing.py: load, normalize, generate mask, compute measurements
        |
        v
  physics_model.py: MRIForwardModel (forward, adjoint, gradient)
        |
        v
  solvers.py: pnp_pgm(model, MSSNDenoiser, y, ...) -> reconstruction
        |
        v
  visualization.py: compute_metrics, plot_comparison, plot_convergence
        |
        v
  output/reconstruction.npy + metrics.json
```
