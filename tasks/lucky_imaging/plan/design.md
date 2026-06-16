## File Structure

```
tasks/lucky_imaging/
├── README.md                  # Problem definition and data description
├── requirements.txt           # Python dependencies
├── main.py                    # Pipeline entry point
├── data/
│   ├── raw_data.npz           # Video frames (1, 101, 960, 1280, 3) uint8
│   └── meta_data.json         # Video metadata and processing parameters
├── plan/
│   ├── approach.md            # Solution methodology
│   └── design.md              # Code architecture (this file)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Frame extraction, mono conversion, blur, brightness
│   ├── physics_model.py       # Quality metrics and image registration
│   ├── solvers.py             # Frame ranking, AP grid, stacking pipeline
│   └── visualization.py       # Plotting utilities and quality metrics
├── notebooks/
│   └── lucky_imaging.ipynb    # End-to-end tutorial notebook
└── evaluation/
    ├── reference_outputs/     # Precomputed stacked images and metrics
    ├── fixtures/              # Per-function test fixtures
    └── tests/                 # Unit and integration tests
```

## Data Flow

```
raw_data.npz (frames)
       │
       ▼
 ┌─────────────┐
 │preprocessing │  extract_frames() → frames_mono() → frames_blurred()
 └─────┬───────┘
       │ mono frames, blurred frames, brightness
       ▼
 ┌─────────────┐
 │physics_model │  frame_quality_laplace() → per-frame quality scores
 └─────┬───────┘
       │ quality scores
       ▼
 ┌─────────────┐
 │  solvers    │  rank_frames() → align_frames_global() → create_ap_grid()
 │             │  → rank_frames_local() → compute_local_shifts()
 │             │  → stack_and_blend() → unsharp_mask()
 └─────┬───────┘
       │ stacked uint16 image
       ▼
 ┌─────────────┐
 │visualization │  plot_frame_comparison(), plot_quality_histogram(), etc.
 └─────────────┘
```

### preprocessing.py

```python
def load_frames(data_dir="data"):
    """Load video frames from raw_data.npz.

    Parameters
    ----------
    data_dir : str -- path to data directory

    Returns
    -------
    frames : ndarray, shape (N, H, W, 3), dtype uint8 -- RGB video frames
    meta : dict -- metadata from meta_data.json
    """

def to_mono(frame):
    """Convert an RGB frame to monochrome (luminance).

    Parameters
    ----------
    frame : ndarray, shape (H, W, 3), dtype uint8

    Returns
    -------
    mono : ndarray, shape (H, W), dtype uint8
    """

def gaussian_blur(frame_mono, gauss_width=7):
    """Apply Gaussian blur and upscale to 16-bit for precision.

    Parameters
    ----------
    frame_mono : ndarray, shape (H, W), dtype uint8
    gauss_width : int -- kernel size (must be odd)

    Returns
    -------
    blurred : ndarray, shape (H, W), dtype uint16
    """

def average_brightness(frame_mono, low=16, high=240):
    """Compute mean brightness within the [low, high] range.

    Parameters
    ----------
    frame_mono : ndarray, shape (H, W), dtype uint8
    low : int -- lower brightness threshold
    high : int -- upper brightness threshold

    Returns
    -------
    brightness : float -- mean brightness of thresholded pixels
    """

def prepare_all_frames(frames, gauss_width=7):
    """Prepare all frames: mono, blurred, brightness, Laplacian.

    Parameters
    ----------
    frames : ndarray, shape (N, H, W, 3), dtype uint8
    gauss_width : int

    Returns
    -------
    frames_data : dict with keys:
        'mono' : ndarray (N, H, W) uint8
        'blurred' : ndarray (N, H, W) uint16
        'brightness' : ndarray (N,) float64
        'laplacian' : ndarray (N, H//stride, W//stride) float32
    """
```

### physics_model.py

```python
def quality_measure_gradient(frame_mono, stride=2):
    """Compute frame quality via finite-difference gradient magnitude.

    Parameters
    ----------
    frame_mono : ndarray, shape (H, W) -- monochrome frame
    stride : int -- downsampling stride

    Returns
    -------
    quality : float -- average gradient magnitude
    """

def quality_measure_laplace(frame_blurred, stride=2):
    """Compute frame quality via Laplacian standard deviation.

    Parameters
    ----------
    frame_blurred : ndarray, shape (H, W), dtype uint16 -- blurred frame
    stride : int -- downsampling stride

    Returns
    -------
    quality : float -- std of Laplacian (higher = sharper)
    """

def quality_measure_sobel(frame_mono, stride=2):
    """Compute frame quality via Sobel gradient energy.

    Parameters
    ----------
    frame_mono : ndarray, shape (H, W) -- monochrome frame
    stride : int -- downsampling stride

    Returns
    -------
    quality : float -- sum of Sobel gradient magnitudes
    """

def multilevel_correlation(reference_box, frame_section, search_width, sub_pixel=True):
    """Two-phase normalised cross-correlation alignment.

    Phase 1: coarsened (stride 2) search over the full search window.
    Phase 2: full-resolution refinement within ±4 pixels of phase-1 result.
    Optional sub-pixel paraboloid fit.

    Parameters
    ----------
    reference_box : ndarray, shape (bh, bw) -- reference patch
    frame_section : ndarray, shape (sh, sw) -- search area in target frame
    search_width : int -- maximum shift in pixels
    sub_pixel : bool -- enable sub-pixel refinement

    Returns
    -------
    shift_y : float -- vertical shift (sub-pixel if enabled)
    shift_x : float -- horizontal shift
    success : bool -- True if correlation peak is valid
    """

def sub_pixel_solve(values_3x3):
    """Fit a 2D paraboloid to a 3×3 grid and find the sub-pixel extremum.

    f(dy, dx) = a*dy^2 + b*dx^2 + c*dy*dx + d*dy + e*dx + g

    Parameters
    ----------
    values_3x3 : ndarray, shape (3, 3) -- correlation values

    Returns
    -------
    dy : float -- sub-pixel vertical correction
    dx : float -- sub-pixel horizontal correction
    """

def phase_correlation(frame_0, frame_1, shape):
    """Compute global shift via FFT phase correlation.

    Parameters
    ----------
    frame_0 : ndarray, shape (H, W)
    frame_1 : ndarray, shape (H, W)
    shape : tuple -- (H, W) for FFT

    Returns
    -------
    shift_y : int -- vertical shift
    shift_x : int -- horizontal shift
    """
```

### solvers.py

```python
def rank_frames(frames_data, method="Laplace", normalize=True, stride=2):
    """Rank all frames by sharpness quality.

    Parameters
    ----------
    frames_data : dict -- output of prepare_all_frames()
    method : str -- "Laplace", "Gradient", or "Sobel"
    normalize : bool -- divide quality by brightness
    stride : int -- downsampling stride for quality computation

    Returns
    -------
    quality_scores : ndarray (N,) -- normalised quality scores (max = 1)
    sorted_indices : ndarray (N,) -- frame indices sorted by descending quality
    """

def find_alignment_rect(reference_frame_blurred, scale_factor=3.0):
    """Find the best patch for global alignment.

    Slides a (H/scale_factor, W/scale_factor) window across the frame,
    scores each location by gradient-based structure metric, returns
    the highest-scoring rectangle.

    Parameters
    ----------
    reference_frame_blurred : ndarray, shape (H, W), dtype uint16
    scale_factor : float

    Returns
    -------
    rect : tuple -- (y_low, y_high, x_low, x_high) of best alignment patch
    """

def align_frames_global(frames_data, sorted_indices, rect, search_width=34,
                         average_frame_percent=5):
    """Globally align all frames to a common reference.

    Parameters
    ----------
    frames_data : dict -- output of prepare_all_frames()
    sorted_indices : ndarray -- frame indices sorted by quality
    rect : tuple -- (y_low, y_high, x_low, x_high) alignment rectangle
    search_width : int -- maximum search radius
    average_frame_percent : int -- % of best frames for mean reference

    Returns
    -------
    shifts : ndarray (N, 2) -- (dy, dx) shift for each frame
    intersection : tuple -- (y_low, y_high, x_low, x_high) common overlap region
    mean_frame : ndarray -- mean of best globally-aligned frames
    """

def create_ap_grid(mean_frame, half_box_width=24, structure_threshold=0.04,
                    brightness_threshold=10):
    """Create a staggered alignment point grid.

    Parameters
    ----------
    mean_frame : ndarray, shape (H, W), dtype int32 -- blurred mean frame
    half_box_width : int -- half-width of each AP box
    structure_threshold : float -- minimum normalised structure (0-1)
    brightness_threshold : int -- minimum brightness to keep an AP

    Returns
    -------
    alignment_points : list of dict -- each with keys:
        'y', 'x' : int -- AP centre coordinates
        'box_y_low', 'box_y_high', 'box_x_low', 'box_x_high' : int -- box bounds
        'patch_y_low', 'patch_y_high', 'patch_x_low', 'patch_x_high' : int -- patch bounds
        'reference_box' : ndarray -- reference patch for correlation
    """

def rank_frames_local(frames_data, alignment_points, frame_percent=10,
                       method="Laplace", stride=2):
    """Rank and select frames independently at each alignment point.

    Parameters
    ----------
    frames_data : dict
    alignment_points : list of dict
    frame_percent : int -- percentage of best frames to select per AP
    method : str -- quality metric
    stride : int

    Returns
    -------
    alignment_points : list of dict -- updated with 'selected_frames' key
    """

def compute_local_shifts(frames_data, alignment_points, global_shifts,
                          search_width=14):
    """Compute local warp shifts at each alignment point.

    Parameters
    ----------
    frames_data : dict
    alignment_points : list of dict -- with 'selected_frames'
    global_shifts : ndarray (N, 2)
    search_width : int

    Returns
    -------
    alignment_points : list of dict -- updated with 'local_shifts' key
    """

def stack_and_blend(frames, alignment_points, global_shifts, intersection,
                     mean_frame, drizzle_factor=1):
    """Stack selected frames per AP and blend into final image.

    Parameters
    ----------
    frames : ndarray (N, H, W, 3), dtype uint8 -- original RGB frames
    alignment_points : list of dict -- with selected_frames and local_shifts
    global_shifts : ndarray (N, 2)
    intersection : tuple -- (y_low, y_high, x_low, x_high)
    mean_frame : ndarray -- globally-aligned mean for background fill
    drizzle_factor : int -- super-resolution factor (1 = off)

    Returns
    -------
    stacked : ndarray, shape (H', W', 3), dtype uint16 -- 16-bit stacked image
    """

def unsharp_mask(image, sigma=2.0, alpha=1.5):
    """Post-processing sharpening via unsharp masking.

    Recovers high-frequency detail suppressed by residual sub-pixel
    misalignment in the stack. Safe to apply because stacking has already
    reduced noise; sharpening a noisy single frame would amplify noise.

    Parameters
    ----------
    image : ndarray, dtype uint16 -- stacked image to sharpen
    sigma : float -- Gaussian blur radius controlling sharpening scale
    alpha : float -- sharpening strength (1.0 = moderate, 2.0 = strong)

    Returns
    -------
    sharpened : ndarray, dtype uint16 -- sharpened image, clipped to [0, 65535]
    """
```

### visualization.py

```python
def plot_frame_comparison(single_frame, mean_frame, stacked_frame, ax=None):
    """Plot side-by-side comparison of single, mean, and stacked images.

    Parameters
    ----------
    single_frame : ndarray -- best single frame (uint8 or uint16)
    mean_frame : ndarray -- simple average of all frames
    stacked_frame : ndarray -- lucky-imaging stacked result
    ax : matplotlib Axes or None

    Returns
    -------
    fig : matplotlib Figure
    """

def plot_quality_histogram(quality_scores, selected_threshold=None, ax=None):
    """Plot histogram of per-frame quality scores.

    Parameters
    ----------
    quality_scores : ndarray (N,) -- normalised quality scores
    selected_threshold : float or None -- draw threshold line
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """

def plot_ap_grid(image, alignment_points, ax=None):
    """Overlay alignment point grid on an image.

    Parameters
    ----------
    image : ndarray
    alignment_points : list of dict with 'y', 'x', 'patch_*' keys
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """

def plot_zoom_comparison(single_frame, stacked_frame, region, ax=None):
    """Show zoomed-in crop comparison.

    Parameters
    ----------
    single_frame : ndarray
    stacked_frame : ndarray
    region : tuple -- (y_low, y_high, x_low, x_high)
    ax : matplotlib Axes or None

    Returns
    -------
    fig : matplotlib Figure
    """

def compute_metrics(stacked, reference_mean, best_frame):
    """Compute quality metrics comparing stacked result to baselines.

    Parameters
    ----------
    stacked : ndarray -- stacked image (uint16)
    reference_mean : ndarray -- simple mean of all frames
    best_frame : ndarray -- single best frame

    Returns
    -------
    metrics : dict with keys:
        'snr_improvement' : float -- SNR gain over single frame
        'sharpness_stacked' : float -- Laplacian variance of stacked
        'sharpness_best_frame' : float -- Laplacian variance of best frame
        'sharpness_mean' : float -- Laplacian variance of mean
        'n_alignment_points' : int
        'n_frames_used' : int
    """
```
