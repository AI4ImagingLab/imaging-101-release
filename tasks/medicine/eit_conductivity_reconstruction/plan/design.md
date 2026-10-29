# Design: EIT Reconstruction Pipeline

## File Structure

```
eit_conductivity_reconstruction/
|-- README.md                  # Problem definition and background
|-- requirements.txt           # Python dependencies
|-- plan/
|   |-- approach.md            # Mathematical formulation and strategy
|   |-- design.md              # This file: architecture and data flow
|-- data/                      # Observation data and metadata
|   |-- raw_data.npz
|   |-- meta_data
|-- src/
|   |-- __init__.py            # Public API re-exports
|   |-- preprocessing.py       # Data loading utilities
|   |-- physics_model.py       # FEM mesh, forward model, protocol
|   |-- solvers.py             # Reconstruction algorithms (BP, GREIT, JAC)
|   |-- visualization.py       # Plotting and metrics
|-- notebooks/
|   |-- reconstruction.ipynb   # Main reconstruction notebook
|-- evaluation/                # Evaluation scripts and saved results
|-- output/                    # Generated figures and reconstructions
```

## Key Function Signatures

### preprocessing.py

```python
def load_observation(data_dir="data") -> dict:
    """Load raw_data.npz and return as a dictionary."""

def load_metadata(data_dir="data") -> dict:
    """Load JSON metadata from the meta_data file."""

def prepare_data(data_dir="data") -> tuple[dict, dict]:
    """Return (observation_dict, metadata_dict)."""
```

### physics_model.py

```python
class PyEITMesh:
    """Wrapper around pyeit mesh creation.
    Attributes: node, element, perm, el_pos
    """

class PyEITAnomaly_Circle:
    """Define a circular anomaly with center, radius, and conductivity."""

class PyEITProtocol:
    """Measurement protocol: excitation and measurement electrode pairs."""

class EITForwardModel:
    """FEM-based forward solver.
    Methods:
        solve_eit(ex_line, perm) -> voltage measurements
    """

def set_perm(mesh, anomaly, background=1.0) -> np.ndarray:
    """Assign conductivity values to mesh elements given anomalies."""

def create_protocol(n_el, dist_exc, step_meas, parser_meas) -> PyEITProtocol:
    """Build adjacent/opposite measurement protocol."""

def sim2pts(pts, tri, perm) -> np.ndarray:
    """Interpolate element-wise values to nodes."""
```

### solvers.py

```python
class BPReconstructor:
    """Back-projection reconstruction.
    Methods:
        setup(mesh, protocol)
        solve(v1, v0) -> ds (element conductivity change)
    """

class GREITReconstructor:
    """GREIT grid-based reconstruction.
    Methods:
        setup(mesh, protocol)
        solve(v1, v0) -> (x_grid, y_grid, ds_grid)
    """

class JACDynamicReconstructor:
    """Jacobian time-difference reconstruction.
    Methods:
        setup(mesh, protocol)
        solve(v1, v0) -> ds (element conductivity change)
    """

```

### visualization.py

```python
def plot_mesh(mesh, ax=None, figsize=(6,6)) -> matplotlib.axes.Axes:
    """Render the FEM triangulation with electrode positions."""

def plot_conductivity(mesh, perm, title="", ax=None) -> matplotlib.axes.Axes:
    """Color-map conductivity on the triangular mesh."""

def plot_greit_image(xg, yg, ds, title="", ax=None) -> matplotlib.axes.Axes:
    """Display GREIT reconstruction as a pixel image."""

def plot_reconstruction_comparison(reconstructions, ground_truths, meshes, metrics=None) -> Figure:
    """Side-by-side ground truth vs reconstruction for all methods."""

def compute_metrics(reconstruction, ground_truth) -> dict:
    """Return {'nrmse': float, 'ncc': float}."""

def print_metrics_table(metrics) -> None:
    """Pretty-print a table of per-method metrics."""
```

## Data Flow

```
                        +----------------+
                        |   raw_data.npz |
                        |   meta_data    |
                        +-------+--------+
                                |
                     load_observation / load_metadata
                                |
                                v
                      +-------------------+
                      | prepare_data()    |
                      | obs dict, meta    |
                      +--------+----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
    +-------------------+            +---------------------+
    | PyEITMesh         |            | PyEITProtocol       |
    | (node, element,   |            | (exc_mat, meas_mat) |
    | el_pos, perm)     |            +----------+----------+
    +--------+----------+                       |
             |                                  |
             +----------------+-----------------+
                              |
                              v
                   +---------------------+
                   | EITForwardModel     |
                   | solve_eit(perm)     |
                   | -> v0 (reference)   |
                   | -> v1 (anomaly)     |
                   +----------+----------+
                              |
              +-------+-------+-------+
              |       |               |
              v       v               v
           +----+  +-----+      +-------+
           | BP |  |GREIT|      |JAC dyn|
           +--+-+  +--+--+      +---+---+
              |       |             |
              v       v             v
         ds_elem  (xg,yg,ds)   ds_elem
              |       |             |
              +---+---+-------------+
                  |       |
                  v       v
       +---------------------+     +-------------------+
       | compute_metrics()   |     | plot_*() functions |
       | nrmse, ncc          |     | comparison figure  |
       +---------------------+     +-------------------+
```

## Design Decisions

1. **Thin wrappers over pyEIT**: The physics_model module wraps pyEIT classes
   and functions to provide a stable interface. This isolates the rest of the
   pipeline from pyEIT API changes.

2. **Solver classes with setup/solve pattern**: Each reconstructor follows the
   same interface -- setup(mesh, protocol) then solve(v1, v0). This makes it
   easy to loop over methods and compare results uniformly.

3. **Separate visualization from computation**: Metrics are computed
   independently of plotting, allowing headless evaluation or batch runs
   without requiring a display.

4. **No hardcoded matplotlib backend**: The visualization module does not call
   matplotlib.use(), leaving backend selection to the caller (notebook, script,
   or GUI).
