# Hidden Scattering Volume Behind a Diffusive Slab

> Recover a three-dimensional relative scattering-response volume from confocal photon arrival histograms blurred by two passages through a strongly scattering slab.
> Domain: physics | Keywords: diffuse optical imaging, inverse scattering, wave migration | Difficulty: Hard

## Background

Strong scattering redistributes photons in direction and arrival time. A short laser pulse passing through a thick, low-absorption layer becomes a broad diffuse impulse. Light returning from objects beyond the layer undergoes a second diffusion process, so a conventional time-of-flight image contains both geometric delay and scattering delay.

Confocal illumination and detection share a scanning location. Time-resolved single-photon counting preserves information about the broad distribution of path lengths, even when ballistic light is too weak for direct imaging. Recovering hidden geometry from these measurements is relevant to remote sensing through obscurants and optical inspection through turbid layers.

## Problem Description

The observed photon counts sample a double-pass diffusion model:

```math
y=M\bigl[\phi*\phi*(A\rho)\bigr]+\eta.
```

Here, y denotes the time-resolved count volume; rho is the unknown relative object response; A models confocal free-space propagation to the object and back; phi is the single-pass slab impulse response; the asterisks denote convolution over time and the two lateral coordinates; M restricts the larger modeled volume to the measured aperture and time window; eta represents counting noise and model error. The symbols y, rho, A, phi, M and eta correspond respectively to the mathematical symbols $y$, $\rho$, $A$, $\phi$, $M$ and $\eta$.

For a point at position r and scan position s, the free-space geometric delay is twice their Euclidean distance divided by the vacuum light speed c0. The diffusion kernel follows the finite-slab diffusion equation with diffusion coefficient D:

```math
D=\frac{1}{3(\mu_a+\mu_s)},\qquad
\frac{1}{v}\frac{\partial\Phi}{\partial t}-D\nabla^2\Phi+\mu_a\Phi=0,
\qquad v=\frac{c_0}{n}.
```

Phi is the diffuse fluence, t is time, mu-a is the absorption coefficient, mu-s is the reduced scattering coefficient, v is light speed within the slab, n is its effective refractive index, and the Laplacian acts over spatial coordinates. These correspond to $\Phi$, $t$, $\mu_a$, $\mu_s$, $v$, $n$ and $\nabla^2$. Extrapolated zero-fluence planes lie one extrapolation length ze outside each slab face; the symbols are defined in the metadata table. The layer is planar and laterally homogeneous. The double convolution is a confocal approximation; it does not describe every nonconfocal path exactly.

The sample is a real scanned reflective object behind a polyurethane scattering layer. Measurements contain512 native time bins at every location of the complete32-by32 aperture. Diffusive blur, finite aperture, missing high frequencies and uncertain absolute timing make recovery ill posed. The supplied native clock is nominal: there is no independent time-origin or absolute range calibration. Use the explicit model-grid convention below; do not interpret the result as a calibrated absolute albedo or surveyed distance.

Save **output/reconstruction.npz** relative to the task workspace, containing exactly one real numeric array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 128, 32, 32) | float32 or float64 | Nonnegative relative scattering-response intensity, arbitrary units; axes batch, nominal depth, lateral x, lateral y. Depth index i is assigned i/127 nominal metres; lateral indices j and k are assigned -0.35+0.7j/31 and -0.35+0.7k/31 metres. Preserve the fixed intensity scale implied by peak-normalized observations; no output normalization or alignment is applied during scoring. |

The final15 nominal depth slices are an excluded boundary-response region and must be zero in the submitted array. Evaluation scores the entire declared volume without cropping, translation or rescaling: cosine NCC (no mean subtraction), RMSE divided by the comparison volume's dynamic range, and maximum absolute error divided by that same range. It also checks nonnegativity and the declared zero boundary convention. These measure agreement with a standard reconstruction, rather than independently measured physical accuracy.

## Data Description

All observation arrays are in **data/raw_data.npz**. No additional array is required.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| photon_counts | (1, 512, 32, 32) | float32 | Measured photon events per native histogram bin and scan point, counts/bin; axes batch, native time, lateral x, lateral y. Entire stored observation volume is retained. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Scan dimensions | 32 by 32 | positions | Full confocal aperture |
| Scan width | 0.7 | m | Each lateral coordinate spans -0.35 to0.35 inclusively |
| Lateral sample spacing | 0.7/31 | m | Uniform scan locations; x is the second spatial array axis, y the third |
| Native bin interval | 16 | ps, nominal | Source-reported nominal histogram interval; not independently calibrated in the supplied array |
| Time origin | 0 at first stored bin | nominal bins | Electronic/optical absolute offset is unavailable |
| Stored time bins | 512 | bins | Full native input |
| Modeled time window | native indices0 through416 inclusive | bins | Supported early window for this reconstruction volume; remaining stored bins remain available |
| Model temporal samples | 128 | samples | Uniform endpoints over the supported window; nominal interval416*16/127ps |
| Vacuum light speed c0 | 300000000 | m/s | Geometric round-trip propagation |
| Effective refractive index n | 1.12 | dimensionless | Separately calibrated scattering-layer model |
| Reduced scattering coefficient mu-s | 262 | 1/m | Layer transport scattering |
| Absorption coefficient mu-a | 0.53 | 1/m | Layer absorption |
| Slab thickness | 0.0254 | m | Diffusive layer |
| Extrapolation length ze | 0.0036 | m | Zero-fluence plane beyond either slab face |
| Source diffusion depth | 1/262 | m | First isotropic source depth into slab |
| Diffusion discretization | 256 by64 by64 | samples | Padded time, x and y; lateral kernel coordinates span -0.7 to0.7 inclusively |
| Diffusion time-grid convention | endpoints0 and4/(c0/n) | s | 256 inclusive samples; its interval58.5621ps differs from the52.4094ps nominal resampled-observation interval; preserve this effective author convention rather than claiming independent clock agreement |
| Discrete boundary | periodic padded diffusion; zero-extended measurements | convention | Observed128-by32-by32 model window occupies the leading block |
| Nominal output depth | 0 to1 inclusive | nominal m | Author reconstruction coordinate convention, not an independently calibrated range |
| Illumination wavelength | 532 | nm | Pulsed laser |
| Pulse repetition | 10 | MHz | Illumination repetition rate |
| Timing response | approximately70 | ps | Combined instrument resolution, distinct from histogram spacing |
| Per-point exposure | 3600/1024 | s | Complete scan exposure3600s |
| Noise | photon counting plus model mismatch | counts | No additional read-noise variance or independently measured background map is supplied |

## Method Hints

A closed-form diffusion deconvolution followed by wave migration is suitable: the homogeneous slab produces a shift-invariant blur, while confocal free-space propagation admits a frequency-domain dispersion relation. Regularized spectral division suppresses frequencies where the diffusion transfer function is small. Migration then maps temporal frequency to depth wavenumber and applies its Jacobian, retaining the positive-frequency branch. The output is squared migrated wave-envelope magnitude, a relative response rather than absolute albedo.

For the declared fixed response convention, use the spectral noise-to-signal regularization1/2500 on globally peak-normalized observations. This fixes the inverse's intensity convention; it is not a measured photon-noise variance. Use the square root of the nonnegative deconvolved response as the wave envelope, without an additional time multiplier.

In the exploding-reflector description the effective free-space wave speed is half the vacuum light speed. If kx, ky and kz are spatial frequencies and nu is temporal frequency, the dispersion relation is:

```math
\nu=\frac{c_0}{2}\sqrt{k_x^2+k_y^2+k_z^2}.
```

Finite windows and diffusion amplify boundary artifacts; use the declared output boundary convention. The model timing discrepancy should be treated as a documented limitation, rather than corrected using knowledge of the hidden object.

## References

Lindell and Wetzstein, *Three-dimensional imaging through scattering media based on confocal diffuse tomography*, Nature Communications11,4517(2020), including supplementary calibration, forward-model and inversion descriptions. [Published paper](https://doi.org/10.1038/s41467-020-18346-3).
