# Complex Shear-Modulus Imaging of an MRE Phantom

> Recover frequency-dependent storage and loss modulus fields from real magnetic-resonance harmonic waves despite noise amplification by spatial differentiation and local-model error.
> Domain: mechanics | Keywords: elasticity imaging, harmonic waves, viscoelastic inversion | Difficulty: Hard

## Background

Mechanical waves carry information about how a material stores and dissipates deformation energy. In magnetic resonance elastography, synchronized motion-encoding gradients convert periodic motion into measurable magnetic-resonance phase. Spatially resolved waves can therefore reveal mechanical properties without applying a local force probe to every image point.

The complex shear modulus combines an elastic storage component with a dissipative loss component. Such imaging supports tissue characterization, quality assessment of soft materials and validation of viscoelastic phantoms. Estimates depend on the excitation frequency and on which wave components are observed.

Recovering modulus requires spatial derivatives of noisy waves. Boundaries, interference, displacement nodes and heterogeneity make direct inversion sensitive to noise and to approximations in the mechanical model. A local scalar Helmholtz model neglects pressure and modulus-gradient terms, so reproducibility alone does not establish absolute mechanical accuracy.

## Problem Description

Recover complex shear-modulus fields for a real multi-inclusion phantom at two excitation frequencies. The supplied observations are the author-preprocessed complex first temporal harmonics of all three motion-encoded channels, over the complete native spatial grid. They are phase-derived wave gradients with a common arbitrary amplitude scale, rather than independently calibrated absolute displacement. For a locally homogeneous incompressible material, a spatial derivative commutes with the scalar wave equation, motivating the same approximate local model for these observations.

For each frequency, use the planar local relation

```math
G(\mathbf r,\omega)\,\Delta_{xy}U_c(\mathbf r,\omega)
+\rho\omega^2U_c(\mathbf r,\omega)\simeq0,
\qquad G=G'+iG'',\quad \omega=2\pi f.
```

Here r is position, f is mechanical frequency in Hz, omega is angular frequency in radians/s, rho is known density in kg/m³, U is the supplied complex harmonic wave, c identifies its three channels, and the planar Laplacian is the sum of second derivatives in x and y with coordinates measured in metres. G, its storage component Gprime and its loss component Gdoubleprime are in Pa. The complex convention uses a nonnegative loss phase, corresponding to G=|G|exp(i phi) with phi in [0,pi]. No z derivative is part of this selected planar model, although every native z plane is reconstructed. A common spatially uniform complex amplitude factor in U cancels from the homogeneous relation. Pressure, out-of-plane curvature and material interfaces can violate it.

Save `output/reconstruction.npz` relative to the task workspace with exactly one array, `reconstruction`, shape `(1, 2, 25, 80, 128)`, dtype `complex128`, axes `(batch, frequency, z, y, x)`, in Pa. Frequencies follow the input order. Preserve both real storage and imaginary loss components, native pixel locations and absolute SI scale. All values must be finite and exactly zero outside `support`. Negative storage estimates may occur in noisy direct estimates; do not silently discard those locations. No complex magnitude conversion, conjugation, fitted global phase, spatial alignment or scaling is applied in evaluation.

Evaluation uses cosine NCC and RMS error divided by the reference dynamic range separately for real storage and imaginary loss at each frequency, on the fixed supplied support. Every quantity/frequency must pass. A separate observation-consistency metric is the L2 norm of G times the regularized planar Laplacian plus rho omega squared times the regularized wave, divided by the L2 norm of rho omega squared times that wave, using all supported voxels and all three channels at each frequency. This is a conditional approximate-model residual, not proof of exact elasticity. No independent measured modulus truth is supplied.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| wave | (1, 2, 3, 25, 80, 128) | complex128 | Real acquisition's phase-derived harmonic wave, arbitrary common amplitude units; axes batch,frequency,channel,z,y,x; channel order z,x,y |
| frequency | (1, 2) | float64 | Mechanical excitation frequencies, in Hz |
| support | (1, 25, 80, 128) | uint8 | Fixed phantom support, dimensionless; one included and zero exterior |
| x | (1, 128) | float64 | Native x coordinates in m |
| y | (1, 80) | float64 | Native y coordinates in m |
| z | (1, 25) | float64 | Native z coordinates in m |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Selected mechanical frequencies | 80, 100 | Hz | Native frequencies retained without resampling |
| Native spacing | 0.0015 in x,y,z | m | Isotropic physical coordinate spacing |
| Density rho | 1000 | kg/m³ | Fixed author material-density convention |
| Channel order | z,x,y | none | Original motion-encoding channel order |
| Temporal samples | 8 | phase samples/cycle | Original phase offsets; first Fourier harmonic retained |
| Harmonic convention | FFT index1, unnormalized forward FFT | none | Native complex first temporal harmonic; no conjugation |
| Phase processing | unwrapped and dejittered source; circular Gaussian smoothing sigma0.65, truncated at3sigma | voxel | Author preprocessing before harmonic extraction |
| Gradient observation | imaginary part of gradient_y(exp(i phase)) times exp(-i phase) | arbitrary wave units | Phase-derived y-gradient for every channel, then harmonic extraction |
| Spatial preprocessing | planar Butterworth low-pass cutoff100, order1 | cycles/m, dimensionless | Applied per native z plane before binding the harmonic observations |
| Common amplitude multiplier | 0.00001 | dimensionless | Uniform author scale; no absolute motion-amplitude calibration is claimed |
| Coordinate origin | x=y=z=0 at first native sample | m | x and y increase with their input indices; display y is upward |
| Spatial boundaries | complete finite native grid, no periodic extension in inversion | none | Derivative regularization uses reflected boundaries; planar model omits z derivatives |
| Support provenance | mean magnitude image, Gaussian sigma0.8, threshold280; fixed interior rectangle x25:103,y15:65 | native index, arbitrary magnitude units | Author geometry support, fixed before inversion; no material labels or fitted-output exclusions |
| Noise | measured experimental noise; variance unavailable | arbitrary wave units | No synthetic noise realization or known variance is assumed |

The data are a processed real acquisition, not simulated waves or known modulus fields. Their common amplitude scale is immaterial to the ideal homogeneous inverse, but processing and the planar approximation remain potential sources of bias. The original dataset was offered as a publicly downloadable sample for local analysis; public redistribution rights were not specified. This local research package does not assert a broader data license.

## Method Hints

The target is the following conditional regularized polar algebraic Helmholtz estimate. Use total-degree quadratic polynomial fits on reflected 9-by-9 planar neighborhoods: smooth the supplied wave first, then apply second-derivative polynomial kernels to that smoothed wave and sum the metre-scaled x/y derivatives. This double filtering and the stated finite-grid conventions define the target; an arbitrary direct derivative estimate is not equivalent.

Regularize the reciprocal Laplacian by separately filtering its magnitude and principal phase with a reflected median footprint whose entries are the outer product of (1,2,3,2,1), retaining entries greater than one. If the relative difference from the median exceeds +3 or -3, replace that value by +3 or -3 times the median, respectively; recombine magnitude and phase and take the reciprocal. Denote the resulting Laplacian by L and the smoothed wave by u. The polar estimate is

```math
A=\frac{\rho\omega^2\sum_c|u_c|}{\sum_c|L_c|+\epsilon},\qquad
\phi=\arccos\left(\frac{-\sum_c(\Re u_c\Re L_c+\Im u_c\Im L_c)}{\sum_c|u_c||L_c|+\epsilon}\right),\qquad
G=A\exp(i\phi).
```

Here A is modulus magnitude in Pa, phi is its phase in radians, Lc is the regularized planar Laplacian of channel c, and epsilon=0.00000001 is the fixed numerical stabilizer in the corresponding denominator units. Use all three channels and the nonnegative-loss arccos branch without an added conjugation or phase alignment. Derivatives use the full measured grid before applying the output support. These fixed choices define a reproducible conditional estimate; compare derivative choices to understand its noise-resolution tradeoff, while retaining both storage and loss.

## References

1. Ragoza,M. and Batmanghelich,K. **Physics-Informed Neural Networks for Tissue Elasticity Reconstruction in Magnetic Resonance Elastography.** MICCAI2023,333–343. [Author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC11141115/). This task selects its algebraic Helmholtz baseline, rather than the paper's neural-network method.
2. BIOQIC, Charité–Universitätsmedizin Berlin. [Public elastography applications and sample data](https://bioqic-apps1.charite.de/). Real unwrapped/dejittered phantom acquisition, processed under the supplied conventions.
