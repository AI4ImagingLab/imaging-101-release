# Staphylococcus Membrane Diffusion Maps

> Recover spatial membrane-marker diffusivity, effective molecular occupancy and correlation offset from a real fluorescence movie whose weak fluctuations are confounded by photobleaching and curved cell geometry.
> Domain: biology | Keywords: fluorescence spectroscopy, molecular diffusion, parameter imaging | Difficulty: Hard

## Background

Molecules move within a cell membrane through thermal diffusion. Their mobility affects membrane organization, transport and the function of associated proteins. Fluorescent membrane markers make this motion observable through fluctuations in light collected from small spatial regions.

Total internal reflection illumination excites a thin layer near the coverslip. This suppresses light from distant cellular structures and helps isolate the lower membrane of a bacterium. Camera acquisition measures many spatial regions simultaneously. Estimating diffusivity from these fluctuations provides a quantitative proxy for the fluidity experienced by the marker, useful for studying temperature adaptation and antimicrobial effects.

Small bacterial membranes have finite size and appreciable curvature. Detection blur, limited photon statistics and changing fluorophore brightness complicate the relation between the recorded movie and molecular mobility. An inferred diffusivity remains conditional on the stated optical and geometric calibration; it is not a direct measurement of viscosity.

## Problem Description

Recover the diffusivity and associated fluctuation parameters of Nile red in the membrane of live *Staphylococcus aureus* at20 degrees Celsius. The observations are the complete native44161-frame fluorescence movie and independently supplied cell outlines in the image plane. The spatial output lattice consists of nonoverlapping2by2camera-pixel apertures. Include an aperture if it contains exactly one positive cell ID; background mixed with that one ID is allowed. Exclude apertures containing two or more different positive IDs. This rule defines52 evaluated apertures and does not depend on the inferred parameters.

For a locally stationary aperture, the normalized intensity covariance at positive lag follows

```math
G(\tau)=\frac{1}{N}\left[\operatorname{erf}(\mu)+\frac{\exp(-\mu^2)-1}{\sqrt{\pi}\mu}\right]^2+c,
\qquad \mu=\frac{a}{2\sqrt{\sigma^2+fD\tau}}.
```

Here intensity is the camera digital count after the known common background convention, lag time is denoted by tau, N is the effective number of fluorescent molecules in the observation area, D is the curvature-conditional diffusion coefficient, c is the dimensionless long-lag correlation offset, a is the aperture side length, sigma is the Gaussian point-spread standard deviation, f is a fixed dimensionless curvature correction and erf is the error function. Optical lengths and lag times use the units below. The covariance is the mean product of centered intensities at two times divided by squared mean intensity, with each lag normalized by its number of overlapping pairs. Zero lag is excluded because of detector/photon noise. At coarser time registers, the measured covariance uses pair-averaged intensities and the original mean normalization.

The mean intensity can drift due to bleaching; the covariance model describes the stationary fluctuations after this drift is accounted for. Weak fluctuations make N and c uncertain, while D and f are not separately identifiable: only their product enters the model. Treat f as a supplied calibration, not a quantity to estimate from this movie.

Save a single file `output/reconstruction.npz` relative to the task workspace with exactly these arrays:

| Output key | Shape | Dtype | Units and convention |
| --- | --- | --- | --- |
| reconstruction | (1,6,40) | float64 | D, square micrometres per second; real, positive in every accepted aperture |
| molecule_number | (1,6,40) | float64 | N, effective molecule count, dimensionless; real and positive in every accepted aperture |
| correlation_offset | (1,6,40) | float64 | c, dimensionless normalized covariance; real, signed |

Axis order is batch,row,column. Each output pixel corresponds to the same2by2input block, without interpolation or registration. All three outputs must be exactly zero outside the fixed accepted apertures; all values must be finite. Physical lengths use the object plane. Values represent aperture-scale estimates, not a super-resolved molecular image.

Evaluation compares all three quantities using uncentered cosine similarity and RMS error divided by each reference quantity's dynamic range, over the fixed accepted apertures only. A separate forward-consistency metric is RMS correlation residual normalized independently by each measured curve's first positive-lag value, aggregated over all52apertures and104lags. All criteria must pass. There is no phase, translation, magnitude, scale or offset alignment. Reference agreement measures reproduction of a validated standard estimator; no independent physical diffusivity truth is available.

## Data Description

The visible `data/raw_data.npz` contains the following arrays. Read with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| frames | (1,44161,12,80) | uint16 | Real fluorescence camera digital counts, ADU; axes batch,time,row,column; all native frames retained |
| cell_labels | (1,12,80) | uint8 | Author-supplied image-domain cell regions, dimensionless IDs0through6; zero denotes background; axes batch,row,column |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Frame interval | 0.0013096382608637214 | s | Exact recorded sampling interval; supersedes the paper's typical acquisition interval |
| Camera pixel side | 0.16 | micrometre | Object-plane square pixel size |
| Output aperture side | 0.32 | micrometre | Nonoverlapping2by2native camera pixels |
| Gaussian PSF sigma | 0.19 | micrometre | Independently calibrated optical blur standard deviation |
| Curvature factor f | 2.1 | dimensionless | Published spherical-cell calibration for approximately0.5micrometre radius; fixed, not inferred from this target |
| Temperature | 20 | degree Celsius | Acquisition condition |
| Marker | Nile red | none | Fluorescent membrane probe |
| Initial transient | frames0through1499 | frame index | Focus-transient region; stationary analysis begins at frame1500 and retains all subsequent42661frames |
| Common camera background convention | minimum of all retained image samples | ADU | Subtract this single publicly computable value from every retained sample |
| Spatial origin | upper-left image corner | micrometre | Raw pixel centres are at(column+0.5,row+0.5)times0.16; output centres use0.32; row increases downward |
| Spatial extent | 12.8by1.92 | micrometre | Widthbyheight, native rectangle; no spatial padding or periodic wrapping |
| Detector | emCCD, Andor iXon897 | none | Paper apparatus; exact shot/read-noise variances are not supplied and must not be assumed known |
| Objective | 100times, NA1.46, oil immersion | dimensionless | Paper apparatus |
| Excitation wavelength | 561 | nm | Paper acquisition protocol |
| Physical approximation | locally free diffusion with fixed curvature correction | none | Finite curved membranes are represented by calibrated effective planar covariance; no exact3D geometry is supplied |

The outlines specify observed cells, not diffusion coefficients or fit-quality selections. They may contain partial-cell boundary apertures. Do not infer absent individual molecular trajectories from these outlines.

## Method Hints

Consider fluorescence correlation spectroscopy with a square-aperture Gaussian detection model and nonlinear least-squares parameter inference. A variance-preserving double-exponential correction can separate a slowly changing mean from faster fluorescence fluctuations. Multi-tau correlation compresses long time series by averaging adjacent samples at successive temporal scales, retaining fine resolution at short lags.

Fit effective occupancy, diffusivity and a long-lag offset jointly. Spatial integration improves statistical reliability at the cost of resolution. Do not apply the curvature correction twice: fitting with fD in the model already returns the corrected D. Quality diagnostics can expose biased curves, but must not change the fixed output support or conceal failed fits.

## References

1. Barbotin,A., Billaudeau,C., Sezgin,E., and Carballido-Lopez,R. **Quantification of membrane fluidity in bacteria using TIR-FCS.** Biophysical Journal123(16),2484–2495(2024). [Published article](https://doi.org/10.1016/j.bpj.2024.06.012).
2. Author experimental dataset, **2024BPJdataset**, Zenodo record11236214. [Data and attribution](https://doi.org/10.5281/zenodo.11236214), Creative Commons Attribution4.0.
