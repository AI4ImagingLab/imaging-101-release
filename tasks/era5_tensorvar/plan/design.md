# Code Design

## File structure

```
era5_tensorvar/
├── README.md
├── requirements.txt
├── main.py                       # End-to-end pipeline + checkpoint download
├── data/
│   ├── raw_data.npz              # 1 sample, batch-first
│   ├── ground_truth.npz          # 1 sample, batch-first
│   └── meta_data.json            # imaging geometry only
├── plan/
│   ├── approach.md
│   └── design.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # data + checkpoint loaders + gdown
│   ├── physics_model.py          # ERA5_K_S, ERA5_K_S_preimage, ERA5_K_O,
│   │                             # transformer blocks, ERA5ForwardModel,
│   │                             # ERA5InverseModel
│   ├── solvers.py                # qp_solver_latent, tensor_var_4dvar
│   └── visualization.py          # metrics + plot_all_channels
├── notebooks/
│   └── era5_tensorvar.ipynb      # loads precomputed reference outputs
└── evaluation/
    ├── metrics.json
    ├── checkpoints/              # populated by download_pretrained_weights
    ├── reference_outputs/        # written by main.py
    ├── fixtures/parity/          # captured upstream intermediates
    └── tests/                    # pytest unit + parity + smoke tests
```

## Function signatures

### `src/preprocessing.py`

```python
PRETRAINED_GDRIVE_ID: str         # Google Drive file ID for ERA5_model_weights.zip
PRETRAINED_FILENAMES: tuple[str]  # ('forward_model.pt', 'C_forward.pt',
                                  #  'inverse_model.pt', 'z_b.pt')

def load_observation(data_dir: str | os.PathLike = "data") -> dict:
    """Read raw_data.npz, return every key as np.ndarray.

    Returned keys (all batch-first, axis-0 length 1):
      'obs_history'      : (1, T, history_len*C, H, W) float32 — normalised obs sequence
      'max_val'          : (1, C) float32                       — per-channel max for de-normalisation
      'min_val'          : (1, C) float32                       — per-channel min for de-normalisation
      'lat_weight_matrix': (1, C, H, W) float32                 — cosine-latitude weighting tiled across channels
    """

def load_ground_truth(data_dir: str | os.PathLike = "data") -> dict:
    """Read ground_truth.npz.
    
    Returned keys:
      'state' : (1, T, C, H, W) float32 — true state sequence in normalised space
    """

def load_metadata(data_dir: str | os.PathLike = "data") -> dict:
    """Parse meta_data.json into a Python dict (no side effects)."""

def select_sample(arrays: dict, sample_index: int = 0) -> dict:
    """Drop the leading batch dim by indexing into it."""

def download_pretrained_weights(weights_dir: str | os.PathLike) -> Path:
    """gdown ERA5_model_weights.zip from upstream and unpack into weights_dir.
    
    No-op if all four checkpoints already exist.
    """

def load_pretrained_models(
    weights_dir: str | os.PathLike,
    device: str | torch.device = "cpu",
) -> tuple[ERA5ForwardModel, ERA5InverseModel, torch.Tensor]:
    """Build the two networks and load their state-dicts. Returns (forward_model, inverse_model, z_b)."""

def default_covariances(hidden_dim: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identity-fallback (B, R, Q): B = 0.1*I, R = I, Q = 0.1*I."""
```

### `src/physics_model.py`

```python
ERA5_SETTINGS: dict
    # state_dim, obs_dim, history_len, seq_length, state_filter_feature_dim,
    # state_feature_dim, obs_feature_dim — fixed by released checkpoints.

# --- Restormer-style transformer blocks (used by Transformer_Based_Inv_Obs_Model) ---
class BiasFree_LayerNorm(nn.Module): ...
class WithBias_LayerNorm(nn.Module): ...
class LayerNorm(nn.Module): ...
class FeedForward(nn.Module): ...
class Attention(nn.Module): ...
class TransformerBlock(nn.Module): ...
class OverlapPatchEmbed(nn.Module): ...
class Upsample_Flex(nn.Module): ...
class Transformer_Based_Inv_Obs_Model(nn.Module):
    def __init__(self, in_channel: int = 50, out_channel: int = 5,
                 LayerNorm_type: str = "WithBias",
                 ffn_expansion_factor: float = 2.66, bias: bool = False,
                 num_blocks: tuple[int, ...] = (2, 2, 2, 2)): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

# --- Tensor-Var feature operators ---
class ERA5_K_S(nn.Module):
    """Convolutional state encoder φ_S : (B, 5, 64, 32) -> (B, 512)."""
    def forward(self, state: torch.Tensor, return_encode_list: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        ...

class ERA5_K_S_preimage(nn.Module):
    """Convolutional state decoder φ_S^{-1} : (B, 512) -> (B, 5, 64, 32).
    
    When `encode_list` is provided, encoder skip features are added at every
    spatial scale (U-Net style)."""
    def forward(self, z: torch.Tensor, encode_list: list[torch.Tensor] | None = None
                ) -> torch.Tensor:
        ...

class ERA5_K_O(nn.Module):
    """Inverse-observation model: (B, history_len*C, H, W) -> (B, C, H, W)."""
    def forward(self, obs: torch.Tensor) -> torch.Tensor: ...

# --- High-level wrappers ---
class ERA5ForwardModel(nn.Module):
    K_S: ERA5_K_S
    K_S_preimage: ERA5_K_S_preimage
    hidden_dim: int  # 512
    seq_length: int  # 10
    C_forward: torch.Tensor | None  # (D, D), assigned externally from checkpoint

    def encode(self, state: torch.Tensor, return_encode_list: bool = True
               ) -> tuple[torch.Tensor, list[torch.Tensor]]: ...
    def decode(self, z: torch.Tensor, encode_list: list[torch.Tensor] | None = None
               ) -> torch.Tensor: ...
    def latent_forward(self, z: torch.Tensor) -> torch.Tensor:
        """z @ C_forward — single feature-space dynamics step."""

class ERA5InverseModel(nn.Module):
    K_O: ERA5_K_O
    K_S: ERA5_K_S          # frozen, present only for state-dict compatibility
    K_S_preimage: ERA5_K_S_preimage  # frozen, present only for state-dict compatibility
    def forward(self, obs: torch.Tensor) -> torch.Tensor: ...
```

### `src/solvers.py`

```python
def qp_solver_latent(
    z_b: np.ndarray,        # (D,)   background latent mean
    seq_z: np.ndarray,      # (T, D) observation features for the window
    F: np.ndarray,          # (D, D) feature-space dynamics matrix
    B: np.ndarray,          # (D, D) background information matrix
    R: np.ndarray,          # (D, D) observation information matrix
    Q: np.ndarray,          # (D, D) process information matrix
    T: int,                 # window length
) -> np.ndarray:            # (T, D) analysed feature trajectory
    """Convex QP for one Tensor-Var assimilation window. Solved with cvxpy."""

def tensor_var_4dvar(
    obs_history: torch.Tensor,             # (T, history_len*C, H, W) on device
    forward_model: ERA5ForwardModel,
    inverse_model: ERA5InverseModel,
    z_b: torch.Tensor,                     # (D,)
    B: np.ndarray, R: np.ndarray, Q: np.ndarray,
    assimilation_window: int,
    total_steps: int,
) -> tuple[np.ndarray, dict]:
    """Run the inverse-obs network, encoder, QP, and decoder for the whole sequence.
    
    Returns
    -------
    trajectory : (total_steps, C, H, W) float32 — decoded analysis in normalised space
    diagnostics : dict with keys
        'inv_obs_seq_z'   : (total_steps, C, H, W) inverse-obs network output per step
        'K_S_seq_z'       : (total_steps, D)       encoder feature per step
        'qp_result'       : (total_steps, D)       analysed latent trajectory
        'evaluation_time_s': float
    """
```

### `src/visualization.py`

```python
CHANNEL_NAMES: tuple[str, ...]  # ('geopotential', 'temperature', 'humidity', 'wind_u', 'wind_v')

def compute_weighted_nrmse_per_channel(
    estimate: np.ndarray,            # (T, C, H, W)
    reference: np.ndarray,           # (T, C, H, W)
    lat_weight_matrix: np.ndarray,   # (C, H, W)
) -> np.ndarray:                     # (C,) latitude-weighted relative L2 error per channel

def compute_metrics_per_channel(
    estimate: np.ndarray, reference: np.ndarray, lat_weight_matrix: np.ndarray,
) -> dict:
    """Returns dict with keys 'ncc', 'nrmse', 'weighted_nrmse' (each (C,) np.ndarray)
    and the corresponding 'ncc_mean', 'nrmse_mean', 'weighted_nrmse_mean' floats."""

def metrics_to_jsonable(metrics: dict) -> dict:
    """Round + cast np.ndarray / np.floating values for JSON serialisation."""

def plot_all_channels(
    estimate: np.ndarray, reference: np.ndarray, metrics: dict | None = None,
    channel_names: Sequence[str] = CHANNEL_NAMES,
):
    """Strip-plot the estimate vs reference for every (channel, timestep) pair.
    Returns a matplotlib Figure. Never calls plt.show or matplotlib.use."""

def print_metrics_table(metrics: dict, channel_names: Sequence[str] = CHANNEL_NAMES) -> None:
    """Pretty-print the per-channel metrics table to stdout."""
```

### `main.py`

```python
_ASS_W: int      = 5
_ASS_T: int      = 5
_SEED: int       = 0
_SAMPLE_INDEX: int = 0

def parse_args() -> argparse.Namespace: ...
def main() -> None:
    """1. download_pretrained_weights → 2. load sample → 3. tensor_var_4dvar
    → 4. compute_metrics_per_channel → 5. save trajectory.npy, ground_truth.npy,
    inv_obs_seq_z.npy, K_S_seq_z.npy, qp_result.npy, metrics.json, comparison.png
    """
```
