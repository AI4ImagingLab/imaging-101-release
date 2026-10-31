# Chromatic Lookup Surface Geometry

> Recover three full surface-depth images from alternating measured color-pattern and white exposures, where weak light and similar calibrated colors make depth ambiguous.
> Domain: mathematics | Keywords: structured light, geometric inference, empirical calibration | Difficulty: Medium

## Background

Projected light encodes surface position in the color observed by a camera. Different depths intercept different parts of the projected pattern. A separate exposure under white illumination helps account for surface color, illumination falloff and camera sensitivity. This provides contact-free measurements of visible surface geometry.

High-speed shape measurements support material testing, robotics and the study of deforming objects. Alternating two exposures permits rapid acquisition, but their appearance must remain sufficiently consistent. Shadows, specular reflections, motion blur and similar colors at different depths limit geometric accuracy.

## Problem Description

The observations are the first three consecutive pattern/white pairs of a real silicone-bunny drop recording. Recover camera-axial depth at every observable pixel of the complete supplied region. Let frame index be $f$, flattened camera-pixel index be $p$, color channel be $c$, calibration sample be $k$, and calibration-factor index be $r$. The measurement model is

```math
q_{f,p,c}=\frac{I^{P}_{f,p,c}}{I^{W}_{f,p,c}},\qquad
\mathbf q_{f,p}=\mathbf C_{p,k_{f,p}}+\boldsymbol\eta_{f,p},\qquad
z_{f,p}=D_{p,k_{f,p}}.
```

Here $I^{P}$ and $I^{W}$ are measured pattern and white intensities, $\mathbf q$ is their dimensionless RGB ratio, $\boldsymbol\eta$ includes noise and model discrepancy, and $z$ is depth in millimetres along the camera's forward axis. The independent planar-board calibration supplies the empirical color dictionary $C$ and its axial-depth dictionary $D$ through factors:

```math
C_{p,k,c}=\sum_{r=1}^{25}L^{C}_{c,p,r}R^{C}_{c,r,k},\qquad
D_{p,k}=\sum_{r=1}^{25}L^{D}_{p,r}R^{D}_{r,k}.
```

The superscripts distinguish color and depth factors. Color-left factors have normalized-ratio units and depth-left factors have millimetre units; right factors are dimensionless. The 749 samples are ordered exactly as supplied and have pixel-dependent calibrated depths. They are not a uniform camera-depth grid. The calibration was acquired independently of the unknown object.

The input-only observation support uses the minimum white intensity across RGB at each pixel. A pixel is supported when that minimum is strictly greater than 0.05 times its per-frame maximum over the image. White normalization assumes approximately unchanged diffuse reflectance between the two exposures; there is no ambient-image subtraction. Unsupported pixels carry no required geometry. Distinct depths can have almost identical colors, so a small photometric residual alone does not establish correct geometry.

Save **output/reconstruction.npz**, relative to the task workspace, with the sole key **reconstruction**. It must be a finite, real **float32** array of shape **(1, 3, 650, 650)**, with axes **batch, chronological frame, sensor row within ROI, sensor column within ROI**. Values are absolute camera-axial depth in **mm**, positive forward. Every unsupported pixel must equal exactly **−1 mm**. Preserve the original sensor orientation: rows increase downward and columns rightward. Do not convert axial depth into radial range, subtract an offset, register, interpolate onto a different grid, fit a scale or clip high-residual estimates.

Each frame's complete observation support is evaluated separately using uncentered cosine NCC, RMS error divided by the reference depth range, and maximum absolute error in mm. All three criteria must pass for every frame. A separate forward check maps each submitted axial depth to its nearest supplied calibrated depth, takes the earliest sample at a tie, and computes the RMS of per-pixel Euclidean RGB residuals over the complete support. This uses the stated calibration directly, without alignment or residual-based masking. Evaluation measures a conditional reconstruction convention rather than independently measured dense geometry.

## Data Description

All public arrays are in **data/raw_data.npz**. RGB channels are ordered red, green, blue. Flattening a pixel means C-order row-major flattening of the 650 × 650 region.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| pattern | (1, 3, 650, 650, 3) | uint16 | Real pattern-exposure intensity in ADU; batch, frame, row, column, RGB. |
| white | (1, 3, 650, 650, 3) | uint16 | Real alternating white-exposure intensity in ADU; same axes as pattern. |
| left_color | (1, 3, 650, 650, 25) | float32 | Independent color-calibration left factors in normalized-intensity-ratio units; batch, RGB, row, column, factor. |
| right_color | (1, 3, 25, 749) | float32 | Dimensionless color-calibration right factors; batch, RGB, factor, ordered calibration sample. |
| left_depth | (1, 650, 650, 25) | float32 | Independent axial-depth calibration left factors in mm; batch, row, column, factor. |
| right_depth | (1, 25, 749) | float32 | Dimensionless depth-calibration right factors; batch, factor, ordered calibration sample. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Native sensor | 1920 × 1080 | pixel | Width × height before the supplied region. |
| Calibrated ROI | (650, 200, 1300, 850) | sensor pixel | Left, top, exclusive right, exclusive bottom; no resizing. |
| Bound frame identifiers | 0, 1, 2 | dimensionless | First three consecutive measured pattern/white pairs. |
| Camera acquisition rate | 900 | exposure/s | Alternating pattern and white flashes. |
| Depth-frame rate | 450 | frame/s | One pattern/white pair per reconstruction. |
| Consecutive pair interval | 1/450 | s | Relative depth-frame times are 0, 1/450 and 2/450. |
| Pattern-to-white interval | 1/900 | s | Two successive exposures are not simultaneous. |
| Exposure duration | 1.0 | ms | Native high-speed acquisition. |
| Detector encoding | 12-bit values stored in uint16 | ADU | Retain measured values rather than rescaling to uint16's maximum. |
| Illumination pattern | GSF_21 | dimensionless identifier | Fixed colored pattern on the analog projector. |
| Independent calibration | Moving planar ChArUco board | — | Does not use the unknown object's shape. |
| Calibration compression rank | 25 per channel | dimensionless | Instrument-table representation, including depth factors. |
| Supplied calibration samples | 749 | count | Use the actual factors; do not replace by the paper's planned 750-step grid. |
| Planned stage stride | 0.2 | mm | Physical calibration-stage translation; not a uniform camera-axial increment. |
| Ambient capture | Unavailable; dark-room acquisition | — | No hidden black image or estimated background is supplied. |
| Support threshold | 0.05 | dimensionless | Fraction of per-frame maximum of the minimum white RGB channel, as defined above. |
| Pixel coordinates | row down, column right | pixel | Full-sensor coordinates are column + 650 and row + 200. |
| Noise parameters | Not independently calibrated | — | No known Gaussian variance or noise-free target is supplied. |
| Spatial boundary | Independent visible camera rays | — | No periodic extension or assumed surface outside the ROI. |

## Method Hints

An empirical inverse can choose the calibrated color nearest to the normalized observation for each camera ray:

```math
\widehat k_{f,p}=\arg\min_k\sum_{c=1}^{3}
\left(C_{p,k,c}-q_{f,p,c}\right)^2,\qquad
\widehat z_{f,p}=D_{p,\widehat k_{f,p}}.
```

Use the earliest ordered sample for an exact tie. The sum is unweighted and the intensity ratios have no fitted gain or offset. Low-rank matrix products and bounded temporary arrays can evaluate the calibration without retaining its entire dense expansion. Spatial or temporal coherence can help disambiguate similar colors, but confidence and geometric accuracy must remain distinct.

## References

- Pereira, Gao, Piadyk, Fouhey, Silva and Panozzo. *LookUp3D: Data-Driven 3D Scanning*. SIGGRAPH Asia 2025. [Paper](https://doi.org/10.1145/3757377.3763986); [complete preprint, version 2](https://arxiv.org/abs/2405.14882v2).
- Pereira et al. *Data of LookUp3D: Data-Driven 3D Scanning*. NYU Data Repository, 2025. [Dataset](https://doi.org/10.58153/1n4p2-ygf77), Creative Commons Attribution 4.0 International.
