# In Vivo Ultrasound Microvascular Mapping

> Reconstruct microvascular density and blood-flow speed from 6000 real contrast-enhanced ultrasound IQ frames despite strong tissue clutter, overlapping bubble responses, and ambiguous temporal associations.
> Domain: Medicine | Keywords: ultrasound microscopy, super-resolution imaging, vascular flow | Difficulty: Hard

## Background

Gas-filled contrast microbubbles scatter ultrasound strongly while circulating through blood vessels. Their acoustic responses extend over several image pixels, even though individual bubbles are much smaller than the acoustic resolution. Accumulating their positions over time makes it possible to resolve vascular structures below the resolution of conventional ultrasound intensity images.

The same circulating scatterers also carry information about flow. Following a bubble across successive ultrafast acquisitions connects spatial localization to motion measurement. This supports investigation of tumor vascular organization, perfusion, and the microcirculation. Tissue echoes, overlapping responses, finite observation time, and incorrect associations can all produce misleading vessel or flow patterns, so spatial and temporal information must be considered together.

## Problem Description

**Benchmark status:** A fair physical acceptance contract is not yet certified: no independent bubble point-response calibration or metric pixel-pitch calibration identifies a unique density/speed field from these beamformed observations. The existing catalogue comparison is a historical method measurement, not biological truth.

The input comprises complex, already beamformed IQ images of an actual mouse mammary tumor after microbubble injection. The benchmark target is the published radial-symmetry/trajectory measurement functional of these observations: reproducible standard-method density and speed maps, rather than independently known anatomical truth. At acquisition-grid coordinate r and time t, a useful measurement model is

```math
Y(r,t)=T(r,t)+\sum_{j=1}^{J(t)} a_j(t)h(r-r_j(t))+\epsilon(r,t).
```

Here Y is the measured complex IQ amplitude in acquisition amplitude units, T is coherent tissue clutter, J is the number of contributing bubbles, a is a complex bubble amplitude, h is the spatial point response, r is a two-dimensional coordinate in acquisition pixels, and epsilon represents noise and model mismatch. Individual responses are locally approximately symmetric in magnitude; an exact calibrated point-response array is not supplied. The unknown bubble positions and associations must be inferred from the observations. Their trajectories determine the requested maps. The benchmark adopts a sparsity convention of at most 200 resolved candidate responses per frame; this is a catalogue definition, not a claim that the true overlapping bubble population is always at most 200. Many bubble configurations can produce similar blurred frames, while tissue and slowly moving bubbles have overlapping temporal content.

There are 30 complete acquisition blocks. Each contains 200 frames. Associations must stop at block boundaries: the blocks are acquired intermittently, rather than as one continuous 6000-frame movie. Retain the complete field of view and every bound frame.

Save **`output/reconstruction.npz`**, relative to the task workspace, with both required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 831, 1281) | float64 | Nonnegative microvascular density, measured as number of retained trajectories crossing each superpixel; axes are batch, axial, lateral |
| speed | (1, 831, 1281) | float64 | Nonnegative mean track speed in acquisition pixel/s; axes are batch, axial, lateral; zero where no retained trajectory crosses the superpixel |

For the output functional, a retained trajectory contains at least nine consecutive observed positions inside one block. Successive positions satisfy a Euclidean displacement of at most one acquisition pixel per frame. Links use the padded complete squared-cost assignment convention defined in Method Hints, retaining only valid original edges, with no missing-frame gap closing. Smooth each coordinate with a centered uniform window: use the largest odd span no greater than min(n,20), where n is the original track length, and shorten the window symmetrically at endpoints. Interpolate linearly at frame abscissae 1+0.08k not exceeding n, for nonnegative integers k. The 0.08 is an increment in original-frame index, not a spatial arc-length interval. Quantize each resulting [axial,lateral,path-speed] row to float32 before grid mapping; raw position tracks used in counting comparisons have the same final quantization.

Both outputs are real arrays. Density is a crossing count, not a normalized probability or a calibrated vessel-volume fraction. A trajectory contributes at most once to any superpixel. Speed is the mean of the contributing trajectories' path speeds, with one equal contribution per trajectory. It is a scalar path speed, rather than an axial velocity component or a vector field. For a trajectory with n original positions, its path-speed convention is interpolated smoothed path length divided by n/500 seconds. This finite-length convention is part of the requested measurement and must not be replaced by an endpoint displacement or a denominator of (n-1)/500.

The output grid uses a subdivision factor of 10. At output zero-based index (i,j), the associated native coordinate is ((i+1)/10-1, (j+1)/10-1) acquisition pixels. Axial coordinates increase downward; lateral coordinates increase rightward. Round a mapped location to its nearest grid index, with halfway values rounded away from zero. Outside-grid locations do not contribute. This includes the original rendering padding and origin shift.

Evaluate both quantities using cosine normalized cross-correlation (NCC, without mean subtraction) and root-mean-square error normalized by the comparison image's dynamic range (NRMSE). There is no fitted gain, spatial alignment, crop, phase alignment, or magnitude conversion during scoring. All four metric conditions must pass. Scoring measures reproducibility of this defined functional, rather than independently verified biological accuracy. Absolute speed in mm/s is not requested: the archived transducer pitch and the example wavelength conversion do not independently establish the pixel spacing of these bound beamformed frames.

## Data Description

The data are real in-vivo observations from the **In Vivo Mouse Tumor** component of the public OPULM/PALA dataset, version 1.1, DOI [10.5281/zenodo.4343435](https://doi.org/10.5281/zenodo.4343435). The bound observations are complete IQ blocks 001 through 030 of a 150-block acquisition, each containing 200 frames. Their spatial dimensions and complex64 samples are preserved without cropping, denoising, synthetic noise, interpolation, or quantization changes. These are beamformed observations, rather than raw channel RF measurements. The bound subset spans the first 30 blocks; incomplete vascular sampling remains a scientific limitation.

`data/raw_data.npz` contains exactly one public array:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| iq | (1, 30, 83, 128, 200) | complex64 | Measured complex IQ amplitude, arbitrary acquisition amplitude units; axes batch, block, axial position, lateral position, frame |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Frame rate within a block | 500 | Hz | Frame interval 0.002 s; do not use this to bridge block gaps |
| Frames per block | 200 | frames | Complete observed time series in each block |
| Block repetition frequency | 2 | Hz | Archived setting; no supplied absolute timestamps, and reported paper duration differs |
| Bound blocks | 001 through 030 | block identifiers | Original chronological file ordering |
| Native image size | 83 axial by 128 lateral | acquisition pixel | No bound spatial crop |
| Coordinate axes | axial down; lateral right | convention | Native pixel centers indexed from zero |
| Transmit frequency | 15 | MHz | Stored transmit/demodulation frequency |
| Receive frequency setting | 60 | MHz | Stored acquisition receive setting |
| Sampling frequency | 30 | MHz | Archived channel sampling rate; input has already been beamformed |
| Compounding steering angles | -10, -5, 0, 5, 10 | degree | Angles used for each compounded IQ image |
| Transmit cycles | 3 | cycles | Stored pulse length setting |
| Transducer element pitch | 0.11 | mm | Element spacing; do not equate it with the beamformed axial/lateral pixel size |
| Native spatial unit | 1 | acquisition pixel | Common coordinate unit used by the selected tracking/rendering convention |
| Absolute image pixel size | unavailable | mm/pixel | No independently verified conversion is supplied |
| Point response | not separately calibrated | acquisition amplitude unit | Local symmetric-response approximation, with no target-derived PSF supplied |
| Noise variance | unknown | amplitude unit squared | No calibrated independent Gaussian-noise variance is asserted |
| Image boundary | finite measured field of view | convention | No periodic spatial boundary or extrapolated frames |
| Output subdivision | 10 | superpixel per acquisition pixel | Fixed output scale; not a claim of independently measured anatomical resolution |

Reference computation promotes the unchanged public complex64 observations to complex128 before temporal subspace decomposition. A single-precision Gram decomposition is sensitive to numerical backend in this data; use adequate numerical precision for the tissue/bubble separation. Tracking interpolation results are quantized to float32 before rendering, preserving the stated output grid conventions.

Dataset records list CC BY 4.0, while accompanying materials identify CC BY-NC-SA 4.0. Treat the bound data and adapted processing as research materials under the more restrictive accompanying terms; attribution and the discrepancy are retained in maintainer provenance.

## Method Hints

Consider temporal-subspace tissue suppression, subpixel localization and trajectory association. These operations trade localization bias, noise rejection and trajectory continuity; count and speed maps depend on the assumed response and detection model. Use observation-derived diagnostics and retain native sampling units when metric calibration is unavailable.

## References

- Heiles, B., Chavignon, A., Hingot, V., Lopez, P., Teston, E., and Couture, O. *Performance benchmarking of microbubble-localization algorithms for ultrasound localization microscopy*. Nature Biomedical Engineering, 2022. DOI: [10.1038/s41551-021-00824-8](https://doi.org/10.1038/s41551-021-00824-8).
- OPULM/PALA real in-vivo dataset, version 1.1. DOI: [10.5281/zenodo.4343435](https://doi.org/10.5281/zenodo.4343435).
- Couture et al. *Ultrasound localization microscopy and super-resolution: A state of the art*. IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 2018. DOI: [10.1109/TUFFC.2018.2850811](https://doi.org/10.1109/TUFFC.2018.2850811).
