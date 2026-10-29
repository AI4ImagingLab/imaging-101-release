# LSRTM Code Design (From-Scratch C-PML)

## Module Structure
```
src/
├── preprocessing.py    # Velocity loading, subregion extraction, migration smoothing
├── physics_model.py    # C-PML forward and Born propagators
├── solvers.py          # Direct-arrival subtraction and LSRTM optimization
└── visualization.py    # Metrics and plotting utilities
```

## Entry-Point Responsibilities
- `main.py` loads batch-first arrays from `data/raw_data.npz` and `data/ground_truth.npz`.
- `main.py` reads acquisition geometry and wavelet/time parameters from `data/meta_data.json`.
- `main.py` strips the leading batch dimension before passing tensors into the solver stack.
- `main.py` writes reconstructed arrays and diagnostic figures to the configured output directory.

## Key Functions
- `preprocessing.load_marmousi(path, ny=2301, nx=751) -> Tensor`
- `preprocessing.select_subregion(v_full, ny=600, nx=250) -> Tensor`
- `preprocessing.make_migration_velocity(v_true, sigma=5.0) -> Tensor`
- `physics_model.make_acquisition_geometry(...) -> (Tensor, Tensor)`
- `physics_model.make_ricker_wavelet(freq, nt, dt, n_shots, ...) -> Tensor`
- `physics_model.forward_model(v, dx, dt, ...) -> Tensor`
- `physics_model.born_forward_model(v_mig, scatter, dx, dt, ...) -> Tensor`
- `solvers.subtract_direct_arrival(observed_data, v_mig, dx, dt, ...) -> Tensor`
- `solvers.run_lsrtm(v_mig, dx, dt, ..., n_epochs=3) -> (Tensor, list)`

## Testing Interfaces
- `evaluation/fixtures/` stores deterministic NumPy fixtures for preprocessing, physics, and solver components.
- `evaluation/tests/` validates numerical parity for deterministic functions and checks output shapes and plotting behavior.
