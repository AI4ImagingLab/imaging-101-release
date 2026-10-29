# Code Design

## File Structure

```
src/
├── __init__.py
├── preprocessing.py     # Data loading + 5-step preprocessing pipeline
├── physics_model.py     # Linear mixing forward model
├── solvers.py           # N-FINDR + FCLS unmixing
└── visualization.py     # Plotting utilities + metrics
```

## Data Flow

```
raw_data.npz ──► load_observation() ──► (volume, axis)
                                            │
                                    preprocess_volume()
                                            │
                                    (processed, proc_axis)
                                            │
                                        unmix()
                                       ╱        ╲
                              endmembers    abundance_maps
                                       ╲        ╱
                              compute_metrics() + save
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict:
    """Returns {'spectral_volume': (40,40,10,B), 'spectral_axis': (B,)}"""

def load_metadata(data_dir: str = "data") -> dict:
    """Returns parsed meta_data.json dict."""

def crop(intensity_data, spectral_axis, region) -> (ndarray, ndarray)
def despike(intensity_data, spectral_axis, kernel_size=3, threshold=8) -> (ndarray, ndarray)
def denoise_savgol(intensity_data, spectral_axis, window_length=7, polyorder=3) -> (ndarray, ndarray)
def baseline_asls(intensity_data, spectral_axis, lam=1e6, p=1e-2) -> (ndarray, ndarray)
def normalise_minmax(intensity_data, spectral_axis, pixelwise=False) -> (ndarray, ndarray)
def preprocess_volume(spectral_volume, spectral_axis) -> (ndarray, ndarray)
```

### physics_model.py

```python
def forward(endmembers: ndarray(K,B), abundances: ndarray(N,K)) -> ndarray(N,B)
def residual(observed, endmembers, abundances) -> ndarray(N,B)
def reconstruction_error(observed, endmembers, abundances) -> float
```

### solvers.py

```python
def extract_endmembers_nfindr(spectral_data: ndarray(N,B), n_endmembers: int) -> ndarray(K,B)
def estimate_abundances_fcls(spectral_data: ndarray(N,B), endmembers: ndarray(K,B)) -> ndarray(N,K)
def unmix(spectral_volume: ndarray(X,Y,Z,B), n_endmembers: int) -> (list[ndarray], list[ndarray])
```

### visualization.py

```python
def compute_ncc(estimate, reference) -> float
def compute_nrmse(estimate, reference) -> float
def compute_metrics(estimate, reference) -> dict
def plot_spectra(spectra, spectral_axis, labels=None, stacked=False) -> Axes
def plot_band_image(volume_band, title="") -> Axes
def plot_abundance_maps(abundance_maps, labels, image_layer=None) -> (Figure, Axes)
def plot_merged_reconstruction(abundance_maps, labels, image_layer) -> Axes
```

## Dependencies

- numpy, scipy (core computation, sparse solvers, optimisation)
- scikit-learn (PCA for N-FINDR dimensionality reduction)
- matplotlib (plotting only)
