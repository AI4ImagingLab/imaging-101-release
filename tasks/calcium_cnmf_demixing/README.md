# Motion-Corrected Calcium Source Demixing

> Reconstruct neuronal footprints, fluorescence, calcium innovations and relative activity from a complete real two-photon movie with tissue motion, overlapping cells and fluctuating background.
> Domain: biology | Keywords: fluorescence imaging, source separation, calcium dynamics | Difficulty: Hard

## Background

Two-photon microscopy uses nonlinear excitation to localize fluorescence near a focal plane inside scattering tissue. Calcium-sensitive fluorescent indicators change their emission as intracellular calcium changes, allowing a microscope to observe the activity of many neurons simultaneously. Fluorescence is an indirect and temporally filtered observation of cellular activity: a transient can persist across many camera frames after its initiating event.

Tissue movement changes the apparent position of a cell, while photons from overlapping structures and spatially diffuse background contribute to the same pixel. Recovering interpretable activity therefore requires estimating spatial footprints, temporal signals and nuisance motion together with the background. These measurements support studies of sensory coding, neural population dynamics and activity-dependent circuit organization. Estimated innovations are fluorescence-model events; they are not independently verified electrical spikes.

## Problem Description

The observation is a single complete two-photon recording. Let a frame index be `t`, a two-dimensional pixel coordinate be `p`, and a source index be `k`. The spatially corrected fluorescence follows a low-rank source-plus-background acquisition model:

```math
Y_t\bigl(p+u_t(p)\bigr)+\beta
=\sum_{k=1}^{K}a_k(p)c_k(t)
+\sum_{q=1}^{2}b_q(p)f_q(t)+e_t(p).
```

Here `Y` is the supplied movie in native fluorescence counts; `u` is the spatial sampling displacement in pixels; `beta` is a known additive intensity offset computed as minus the minimum of the first 400 frames; `a` and `b` are nonnegative dimensionless neuronal and background spatial factors; `c` and `f` are their temporal coefficients in fluorescence counts; `e` represents noise and unmodeled fluorescence. The inverse displacement used to resample a frame is related to the reported image-content correction shifts by a sign change. Report shifts using the correction convention defined in Data Description. Residual interpolation errors and model mismatch remain possible.

Calcium dynamics can be described by:

```math
c_k(t)=d_k+v_k(t),\qquad
v_k(t)=g_k v_k(t-1)+s_k(t),\qquad 0\le g_k<1,\quad s_k(t)\ge0.
```

The constant `d` is a component baseline, `v` is its calcium transient, `g` is a dimensionless decay factor per frame and `s` is a nonnegative innovation in fluorescence counts per frame. Temporal coefficient `c` includes the fitted baseline and initial-condition contribution. Background temporal coefficients may be signed. Spatial overlap, correlated activity, unknown source count, spatial scale ambiguity and slow background changes make the factorization ill posed. Motion must be estimated from the observations without a supplied registered movie or neuron templates.

Save `output/reconstruction.npz` relative to the task workspace. All arrays must be finite real `float32` or `float64`, batch first, with these exact keys:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, K, 170, 170) | float32 or float64 | Nonnegative source footprints, axes batch/source/row/column; dimensionless, each footprint has spatial L2 norm 1 |
| fluorescence | (1, K, 3000) | float32 or float64 | Source fluorescence including fitted baseline, axes batch/source/time; native fluorescence counts |
| innovations | (1, K, 3000) | float32 or float64 | Nonnegative AR innovations, same source order; fluorescence counts per frame |
| dff | (1, K, 3000) | float32 or float64 | Relative fluorescence change after local baseline/background correction; dimensionless |
| background_spatial | (1, 2, 170, 170) | float32 or float64 | Nonnegative background spatial factors with unit spatial L2 norm; dimensionless |
| background_temporal | (1, 2, 3000) | float32 or float64 | Matching background temporal coefficients; native fluorescence counts |
| rigid_shifts | (1, 3000, 2) | float32 or float64 | Global image-content correction shift, final axis row/column; pixels |
| nonrigid_shifts | (1, 3000, 9, 2) | float32 or float64 | Local image-content correction shifts, final axis row/column; pixels |

`K` is the number of accepted sources and must lie between 1 and 196. Every source-indexed array must use the same `K` and ordering. A solver may choose its own source ordering. Footprints are compared after a fixed Hungarian assignment maximizing total spatial cosine similarity, and only pairs with cosine at least 0.7 count as matched. Unmatched sources reduce detection F1, defined as twice the number of matches divided by the sum of the two source counts. No spatial translation, temporal shift or intensity rescaling is fitted in evaluation. Require detection F1 at least 0.95; matched footprints and fluorescence require cosine at least 0.98, innovations at least 0.95 and relative activity at least 0.98. Dynamic-range-normalized RMS error boundaries are 0.015 for footprints and 0.025 for the three temporal quantities.

Innovations are nonnegative up to an absolute numerical tolerance of 0.0001 fluorescence count; float32 AR subtraction can produce smaller negative roundoff values, which are retained rather than clipped. Spatial factors permit negative roundoff no larger than 0.0000001.

Background factors have the unit spatial gauge specified above; their full rank-two product is evaluated, allowing permutation and equivalent changes of factor basis. Its cosine must be at least 0.99 and normalized RMS error at most 0.025. Both reported motion fields must agree within 0.11 pixel. Cosine uses flattened vectors without mean subtraction. RMS error is divided by the reference dynamic range. Agreement assesses a validated standard method and does not establish physiological ground truth.

Relative fluorescence is defined using source signal plus its projected residual, a running baseline and projected background. The residual uses the full fitted source model before quality pruning. Project this movie residual onto each unit-norm accepted spatial footprint and add its fluorescence coefficient to obtain signal `F`; project the fitted background onto the same footprint to obtain its background trace. Choose a percentile at the cumulative probability of the density mode of the first 250 signal samples, using a diffusion kernel-density estimate. Apply that percentile to signal and background with a 250-frame running percentile filter and reflected boundaries, yielding signal baseline `L` and background baseline `B`. Report `(F-L)/(B+L)`. The scale is relative change, not percent. These normalization conventions define the reported activity quantity; they are not acquisition calibration or hidden physiological labels.

## Data Description

`data/raw_data.npz` contains the complete real recording, without spatial cropping, frame removal or new simulated observations.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| movie | (1, 3000, 170, 170) | float32 | Native unregistered fluorescence, axes batch/time/row/column; instrument fluorescence counts in arbitrary intensity units |

The recording is the public Sue two-photon movie credited by its providers to Sue Ann Koay and David Tank, Princeton University. Negative native intensities are present; they must not be silently clipped before defining the intensity offset. Absolute photon calibration is unavailable.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Frames | 3000 | frame | Full temporal extent, 100 seconds nominally |
| Frame rate | 30 | Hz | Nominal rate distributed with the reference acquisition configuration |
| Frame dimensions | 170 by 170 | pixel | Rows increase downward, columns increase rightward |
| Pixel pitch | 2 by 2 | micrometre/pixel | Nominal metadata; not independently instrument calibrated |
| Indicator decay time | 0.4 | s | Nominal transient timescale supplied with the acquisition configuration |
| Native intensity | arbitrary signed fluorescence counts | count | No absolute photon conversion available |
| Intensity offset | minus minimum of frames 0 through 399 | count | Observation-derived offset, no target dependence |
| Rigid search support | row and column displacement from -6 to +6 | pixel | Admissible global motion support |
| Nonrigid grid windows | starts 0, 50, 96 along each axis; window size 74 | pixel | Nine overlapping windows; enumeration row-start outer loop, column-start inner loop |
| Nonrigid deviation support | plus or minus 3 from the rigid estimate | pixel | Local displacement support |
| Motion precision | 0.1 | pixel | Subpixel Fourier registration resolution |
| Motion sign | positive moves image content down/right | pixel | Image-content correction, row then column; source sampling has opposite sign |
| Motion boundaries | copy nearest valid edge | none | Cubic warp with replicated boundary values; no periodic image wrap |
| Background rank | 2 | component | Diffuse low-rank fluorescence contribution |
| Noise | unknown, estimated per pixel/trace | count RMS | Frequency band 0.25 to 0.5 cycles per frame is available for stationary noise estimation |

## Method Hints

A suitable family combines piecewise rigid subpixel image registration with constrained nonnegative matrix factorization. Overlapping spatial patches can initialize neuronal footprints and background factors, with duplicate sources merged using spatial overlap and temporal correlation. Global spatial and temporal refitting then separates overlapping sources while retaining the full recording.

Spatial sparse nonnegative regression and nonnegative AR deconvolution serve complementary roles: spatial constraints localize individual sources, while the calcium model rejects temporal fluctuations inconsistent with indicator dynamics. An active-set pool method can solve the AR noise-constrained innovation problem explicitly. Source quality should consider both temporal evidence and spatial agreement with observed transient frames. These criteria identify reconstructable components rather than providing an independently labeled cell census.

## References

1. Giovannucci et al. CaImAn: an open source tool for scalable calcium imaging data analysis. eLife, 2019. DOI: https://doi.org/10.7554/eLife.38173.
2. Pnevmatikakis et al. Simultaneous denoising, deconvolution, and demixing of calcium imaging data. Neuron, 2016. DOI: https://doi.org/10.1016/j.neuron.2015.11.037.
3. Pnevmatikakis and Giovannucci. NoRMCorre: an online algorithm for piecewise rigid motion correction of calcium imaging data. Journal of Neuroscience Methods, 2017. DOI: https://doi.org/10.1016/j.jneumeth.2017.07.031.
4. Friedrich, Zhou and Paninski. Fast online deconvolution of calcium imaging data. PLOS Computational Biology, 2017. DOI: https://doi.org/10.1371/journal.pcbi.1005423.
5. Provider movie download: https://users.flatironinstitute.org/~neuro/caiman_downloadables/Sue_2x_3000_40_-46.tif.
