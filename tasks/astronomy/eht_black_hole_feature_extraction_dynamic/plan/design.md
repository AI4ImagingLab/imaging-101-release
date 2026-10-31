# Code Architecture: Dynamic Feature Extraction

## Module Overview

```
src/
├── generate_data.py      # Time-varying crescent + EHT 2017 observation simulation
├── preprocessing.py      # Per-frame UVFITS loading + closure extraction
├── physics_model.py      # SimpleCrescentParam2Img + NUFFTForwardModel (from feature_extraction)
├── solvers.py            # AlphaDPISolver + RealNVP flow (from feature_extraction)
└── visualization.py      # Ridge plots, parameter evolution, frame montage
```

## Key Functions

### `generate_data.py`

```python
generate_simple_crescent_image(npix, fov_uas, diameter_uas, width_uas, asymmetry, pa_deg)
    → (npix, npix) ndarray

generate_dynamic_dataset(n_frames=10, obs_duration_hr=8.0, tstart_gmst=7.5,
                          pa_start_deg=-130, pa_end_deg=-50, ...)
    → dict (metadata)
    # Saves: data/obs_frame_XX.uvfits, data/gt.fits, data/meta_data
```

### `preprocessing.py`

```python
prepare_frame(uvfits_path, npix, fov_uas)
    → (obs_data, closure_indices, nufft_params, flux_const)
    # Per-frame equivalent of prepare_data() from feature_extraction

load_metadata(data_dir) → dict
load_observation_from_uvfits(uvfits_path) → dict
extract_closure_indices(obs) → dict
compute_nufft_params(obs, npix, fov_uas) → dict
estimate_flux(obs) → float
```

### `physics_model.py` (copied from feature_extraction)

```python
SimpleCrescentParam2Img(npix, fov, r_range, width_range)
    # 4-param geometric model: [r, sigma, asymmetry, PA] → (npix, npix) image

NUFFTForwardModel(ktraj_vis, pulsefac_vis)
    # Image → visibilities via NUFFT + pulse correction
```

### `solvers.py` (copied from feature_extraction)

```python
AlphaDPISolver(npix, fov_uas, n_flow, geometric_model='simple_crescent', ...)
    .reconstruct(obs_data, closure_indices, nufft_params, flux_const) → dict
    .importance_resample(obs_data, closure_indices, nufft_params, n_samples) → dict
    .extract_physical_params(p) → ndarray  # [diameter, width, asymmetry, PA]

RealNVP(nparams, n_flow, affine, seqfrac, permute, batch_norm)
    # Normalizing flow: z ~ N(0,I) → x ∈ R^nparams
```

### `visualization.py`

```python
plot_ridge(params_per_frame, param_names, gt_per_frame, weights_per_frame,
           frame_times, save_path=None)
    → Figure  # Ridge plot (Figure 13 style)

plot_param_evolution(params_per_frame, param_names, gt_per_frame,
                      weights_per_frame, frame_times, save_path=None)
    → Figure  # Mean ± 1σ error bar plot

plot_frame_images(images, frame_times, pixel_size_uas, gt_images, save_path)
    → Figure  # Posterior mean vs ground truth montage

compute_frame_metrics(params_per_frame, gt_per_frame, weights_per_frame, param_names)
    → dict  # Per-frame means, stds, biases
```

### `main.py`

```python
main()
    # 1. Load metadata
    # 2. Loop over frames: prepare_frame → AlphaDPISolver → importance_resample
    # 3. Generate ridge plot + param evolution + frame images
    # 4. Compute and print metrics
    # 5. Save all outputs
```

## Data Flow

```
generate_data.py
    SimpleCrescent(PA rotating) → ehtim.observe(EHT2017) → obs_frame_XX.uvfits

main.py (per frame):
    obs_frame_XX.uvfits → prepare_frame() → (obs_data, closure_indices, nufft_params)
    → AlphaDPISolver.reconstruct() → trained flow
    → AlphaDPISolver.importance_resample() → (params_physical, weights, images)

    All frames → plot_ridge() → ridge_plot.png
```
