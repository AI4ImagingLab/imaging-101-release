# pMHC Single-Molecule Feature Localization

> Recover subpixel locations and effective intensity moments of sparse fluorescent pMHC features from noisy native microscopy frames whose blur and camera baseline obscure their centers.
> Domain: biology | Keywords: fluorescence microscopy, feature localization, moment inversion | Difficulty: Medium

## Background

Fluorescent labels reveal molecules through a diffraction-broadened optical image. A bright spot spans several camera pixels even when the emitter is much smaller than a pixel. Its intensity distribution carries information about the emitter center, while photon fluctuations, camera noise and spatially varying background perturb that distribution.

Subpixel localization is used to measure molecular organization and to construct trajectories in cell biology. Isolated spots permit more precise center estimates than the optical spot width, but this requires a defined background and feature-size convention. Overlapping or dim emitters and image boundaries can produce missed or biased detections. Pixel coordinates alone do not establish a physical resolution without camera magnification calibration.

## Problem Description

The complete author-supplied movie contains ten AF647/pMHC frames. A local fluorescence image can be represented as

```math
I_t(u,v)=b_t(u,v)+\sum_{k=1}^{N_t}m_{tk}h_{tk}(u-x_{tk},v-y_{tk})+\eta_t(u,v).
```

Here $t$ is the frame index; $(u,v)$ are integer camera-pixel centers with $u$ rightward and $v$ downward; $b$ is smooth background in analog-to-digital units (ADU); $N$ is the unknown number of visible features; $(x,y)$ are subpixel centers; $h$ is a normalized optical spot profile with units pixel^-2; $m$ is integrated intensity in ADU; and $\eta$ is measured camera/fluorescence noise in ADU. No fixed instrument PSF, photon gain, noise variance, exposure time, or physical pixel size is supplied. Recover the **effective bandpass-feature moments** specified below, rather than absolute photon counts or molecular copy numbers. Unknown background, finite apertures and overlapping spots make these effective estimands differ from a unique noiseless object.

Save `output/reconstruction.npz` relative to the task workspace with key `reconstruction`, a finite real `float64` table of shape `(1, N, 6)`. Here $N$ is the inferred total number of retained features across the ten frames and may be zero. Row order is unrestricted. Columns are `frame, x, y, mass, size, ecc`: frame is an integer-valued index0–9; x/y are pixel coordinates in `[0,128)` with origin at the upper-left pixel center; mass is the positive integrated bandpass-feature intensity in ADU; size is its positive effective original-estimator size in pixels; ecc is its nonnegative dimensionless second-angular-moment eccentricity. These are effective image-feature measurements; mass is not photon count and size is not calibrated optical resolution. Return all six columns for every retained feature. The method convention defining these moments is described in Method Hints.

Evaluation matches features only within the same frame using a0.5-pixel distance gate, without translation or scale fitting. Detection F1 penalizes missing/extra features; matched position, relative mass, size and eccentricity errors are assessed separately. Every missing reference feature contributes unit squared loss to each error, so an empty or incomplete table cannot evade evaluation. This tests agreement with a standard estimator on measured data, not independent biological ground-truth accuracy.

## Data Description

`data/raw_data.npz` contains the entire native author movie without cropping, downcasting or target-derived calibration.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frames | (1, 10, 128, 128) | uint16 | Ten complete fluorescence frames in native camera ADU; axes batch, frame, downward y, rightward x. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample identifier | pMHC_AF647_200k_000_ | — | Author's original acquisition identifier; no additional biochemical conditions are supplied. |
| Frame count | 10 | frames | Entire provided movie, native order0–9. |
| Field size | 128 by128 | pixels | Entire native camera field; all frames retained. |
| Pixel coordinates | integer centers0–127 | pixels | x increases right, y increases down. |
| Intensity encoding | unsigned16bit | ADU | Native stored camera values; no photon-gain conversion. |
| Physical pixel spacing | unavailable | — | Do not infer nanometer distances. |
| Exposure/frame interval | unavailable | — | Frame index is not a calibrated time axis. |
| Camera noise model | unspecified | ADU | No calibrated variance or Poisson gain is supplied. |
| PSF calibration | unavailable | — | Feature widths are inferred image moments, not supplied target-derived calibration. |
| Spatial boundary | finite detector field | pixels | Features lacking a complete moment/interpolation aperture are excluded by the standard estimator. |

Data were released by Lukas Schrangl under CC0-1.0.

## Method Hints

A Crocker–Grier bandpass followed by local maximum detection and two-pass subpixel moment refinement is suitable for isolated Gaussian-like spots. The effective estimator uses a Gaussian-minus-box kernel with feature radius3pixels and noise correlation radius1pixel, a circular moment aperture of radius3.5pixels, and positive-part filtered intensities. Keep filtered maxima at least300ADU and initial aperture mass strictly above5000ADU. These fixed conventions define the image-feature estimands; they are not additional instrument calibration.

For local nonnegative intensities $J(u,v)$ and circular mask $W$, define mass $M=\sum JW$, centroid $(\sum uJW,\sum vJW)/M$, and the effective original size R by the following implementation convention. With S the squared radial pixel offset inside the centered square ROI, the original estimator uses: 

```math
R^2=\frac{\sum_{\mathrm{square\ ROI}} J(SW+1/6)}{M}.
```

Here the additive pixel-area term acts on the entire square ROI, including pixels outside the circular mask. Thus R is a radius-like effective original estimator, rather than the strict central second moment of only the circular aperture. Applying the area correction only to included pixels would instead give sum(J S W)/M + 1/6; this alternative is not the selected estimator. A bilinear recentering and second moment pass reduces pixel-locking. The eccentricity is the magnitude of the second angular moment of the centered aperture, normalized by its noncentral mass. Preserve all inferred moments instead of interpreting the peak pixel as integrated intensity. A known synthetic Gaussian and an integer translation can test coordinate signs and moment scaling without using hidden observations.

## References

- Crocker and Grier, “Methods of Digital Video Microscopy for Colloidal Studies.” Journal of Colloid and Interface Science179,298–310(1996). https://doi.org/10.1006/jcis.1996.0217.
- Gao and Kilfoil, “Accurate detection and complete tracking of large populations of features in three dimensions.” Optics Express17,4685–4704(2009). https://doi.org/10.1364/OE.17.004685.
- Schrangl, sdt-python documentation and native AF647/pMHC example data.  (code BSD-3-Clause; observations CC0-1.0).
