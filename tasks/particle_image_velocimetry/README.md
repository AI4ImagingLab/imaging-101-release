# Particle Image Velocimetry

> Reconstruct a spatially varying two-component tracer velocity field from paired particle images despite finite-window ambiguity and subpixel displacements.
> Domain: mechanics | Keywords: particle imaging, velocity estimation, correlation imaging | Difficulty: Medium

## Background

Particle image velocimetry observes small tracer particles illuminated within a fluid. Light scattered by the tracers provides a visible pattern whose motion between exposures approximates fluid motion, provided particles follow the flow and remain within the illuminated region. A camera records the intensity pattern rather than directly measuring velocity.

The technique provides nonintrusive spatial measurements in laboratory fluid mechanics, including shear flows, wakes, mixing and transport. Limited particle density, illumination variation, motion out of the imaging plane and finite image resolution restrict how reliably particle motion can be inferred. The recovered planar motion describes the projected tracer transport and does not determine unobserved out-of-plane velocity.

## Problem Description

For each local image neighborhood, let A(r) and B(r) be the two measured intensities at image coordinate r=(x,y), with x increasing right and y increasing down. A locally constant displacement d approximately satisfies

```math
B(\mathbf r) = A(\mathbf r-\mathbf d)+\eta(\mathbf r),\qquad
\mathbf v=\mathbf d/\Delta t.
```

Here eta represents intensity changes, noise and particles entering or leaving a local neighborhood; v=(u,v) is the horizontal and vertical velocity, and Delta t is the interval between frames. The tracer image itself can be viewed as a sum of particle point-spread functions weighted by scattered intensity. Correspondence is ambiguous because particles are visually similar and only a finite neighborhood is visible. Local shear also limits the constant-displacement approximation.

The paired repository observations form one imaging case. Recover the velocity at the fixed centers of the acquisition analysis grid described below. There is no supplied physical length or clock calibration: report motion in image pixels per frame interval, with no claim of SI flow speed. Save `output/reconstruction.npz` relative to the task workspace, containing all of:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 7, 10, 2) | float64 | Real signed velocity in pixel/frame interval; axes batch, grid row, grid column, component; component 0 is rightward u, component 1 is downward v. |
| confidence | (1, 7, 10) | float64 | Dimensionless peak-to-second-peak ratio on each normalized correlation map, with rejected peaks assigned zero as defined below. |

Every value must be finite. The velocity scale and signs are fixed; evaluation applies no rotation, sign change, translation, magnitude conversion or fitted rescaling. Evaluation uses cosine NCC without mean subtraction and dynamic-range-normalized RMS error jointly over signed velocity components, plus relative L2 error on confidence. All required metrics must pass. Agreement measures consistency with a validated standard method and is not independent proof of physical accuracy.

## Data Description

The two images are supplied particle-image examples from the OpenPIV software distribution, preserved without resizing or generated blur. The distribution calls its example a real PIV demonstration, but it does not establish the apparatus, original acquisition date or calibrated physical frame spacing. Treat these as repository-provided observations with incomplete experimental provenance.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frame_a | (1, 369, 511) | uint8 | First monochrome particle image, digital counts from 0 to 255; axes batch, row, column. |
| frame_b | (1, 369, 511) | uint8 | Second monochrome particle image, digital counts from 0 to 255; axes batch, row, column. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Image origin | top left | pixel | x is column, y is row; both positive toward increasing array index. |
| Spatial calibration | 1 | image pixel/pixel | No physical length calibration is available. |
| Frame spacing | 1 | frame interval | Relative timing convention; elapsed seconds are unavailable. |
| Search support | 64 by 64 | pixel | Fixed neighborhood defining each output location and correlation support. |
| Interrogation support | 32 by 32 | pixel | Central square of frame A in the 64-square neighborhood. |
| Neighbor overlap | 16 | pixel | Search windows overlap by this amount, giving grid step 48. |
| Grid x centers | 32 + 48 j, j=0,...,9 | pixel | Centers of actual extracted windows, without shifting the image or recentering unused margins. |
| Grid y centers | 32 + 48 i, i=0,...,6 | pixel | Centers of actual extracted windows. |
| Intensity response | stored digital counts | count | No radiometric calibration or sensor-noise distribution is supplied. |
| Boundary convention | circular within each 64-square map | pixel | Periodic correlation; no image-edge padding; unmatched trailing margins are unused. |
| Confidence exclusion | square half-width 2 | correlation pixel | Exclude the 5 by 5 neighborhood centered on the first peak when finding the second. |
| Peak stability offset | 0.0000001 | normalized correlation | Added to each correlation map after finding an interior peak, before subpixel fitting and confidence measurement. |

The confidence observable is the maximum correlation divided by the largest correlation outside the exclusion square. An edge maximum or first-peak height below 0.001 has confidence zero. A zero second peak, or a second peak on the edge whose height exceeds half the first, also yields zero. These conventions define the requested auxiliary quantity, not a calibrated probability of correctness.

## Method Hints

Use local cross-correlation with a larger search region in the second image. Per-window mean removal and standard-deviation normalization suppress illumination variation. A central support mask in the first window allows a smaller particle interrogation region while retaining a larger displacement search range. Preserve the fixed circular boundary convention when evaluating correlation.

The correlation theorem gives a useful computational form:

```math
C(\boldsymbol\delta)=\frac{1}{S^2}\operatorname{fftshift}
\mathcal F^{-1}\!\left[\overline{\mathcal F(\widetilde A)}\,
\mathcal F(\widetilde B)\right],\qquad S=64.
```

Here F is the discrete Fourier transform, the overbar denotes complex conjugation, and the tildes denote population-standardized window intensities. First standardize both search windows, mask the central support in A, then standardize the masked A and the B window before correlation. Subtract the central correlation coordinate from the peak location to obtain the signed displacement. A three-point Gaussian fit around the maximum gives subpixel motion; use a parabolic fit if any fitted intensity is negative. Local peak confidence exposes ambiguous matches. No training or global flow regularization is needed for this standard family.

## References

- R. J. Adrian, *Particle-Imaging Techniques for Experimental Fluid Mechanics*, Annual Review of Fluid Mechanics 23, 261-304 (1991). https://doi.org/10.1146/annurev.fl.23.010191.001401
- OpenPIV contributors, *OpenPIV: an open-source particle image velocimetry software*, software archive. https://doi.org/10.5281/zenodo.593157
- OpenPIV documentation: https://openpiv.readthedocs.io/

Adapted software and redistributed example images retain the distribution's GPL-3.0-or-later notices; see the consolidated original licenses and attribution in `test/validation.md`.
