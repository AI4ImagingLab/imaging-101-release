# Microwave Dielectric Contrast Imaging

> Reconstruct a spatial dielectric-contrast image from sparse complex microwave scattering measurements despite multiple scattering, limited angular sampling and ill-conditioned field inversion.
> Domain: physics | Keywords: inverse wave scattering, dielectric tomography, nonlinear imaging | Difficulty: Hard

## Background

Electromagnetic waves change phase and amplitude as they interact with materials of different electrical permittivity. Measuring the scattered field around a region provides indirect information about its internal dielectric structure. Microwave imaging uses this effect to inspect inaccessible material distributions without opening the region.

At appreciable contrast, the illuminating wave is itself distorted inside the object. The scattered signal therefore depends on both the unknown material and the unknown internal field, producing a nonlinear inverse problem. Sparse receivers, finite wavelength and measurement noise further limit spatial resolution. Applications include nondestructive inspection, subsurface sensing and electromagnetic characterization of dielectric structures.

## Problem Description

Recover the two-dimensional real relative-permittivity contrast of a lossless isotropic region in a known homogeneous background. The supplied observations are existing author-simulated complex scattered electric fields for multiple plane-wave illuminations; no new synthetic observations were generated for this task. The scalar transverse-magnetic model assumes invariance along z. With time convention exp(+i omega t), the field equations are


```math
E(r,s)=E^{\mathrm i}(r,s)+k_b^2\int_D g(r,r')\chi(r')E(r',s)\,dA',\qquad
E^{\mathrm s}(r_m,s)=k_b^2\int_D g(r_m,r')\chi(r')E(r',s)\,dA',
```


```math
g(r,r')=-\frac{i}{4}H_0^{(2)}(k_b|r-r'|),\qquad
\chi=(\epsilon_r-\epsilon_b)/\epsilon_b,\qquad
E^{\mathrm i}(r,s)=\exp[-ik_b(x\cos\phi_s+y\sin\phi_s)].
```


Here r=(x,y) and r' are spatial coordinates in meters, D is the image region, s the illumination index, r_m the receiver position, E the unknown total electric field, E_i the known incident plane wave, E_s the observed scattered field, k_b the background wavenumber, epsilon_r the unknown relative permittivity, epsilon_b its known background value, phi_s the propagation angle, H the Hankel function and i the imaginary unit. Fields are normalized by the unit incident amplitude. Unknown contrast and total field multiply inside the integral; treating E as the incident field alone neglects multiple scattering. The sparse complex measurements do not uniquely determine every spatial cell without regularization.

Save `output/reconstruction.npz` with:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 30, 30) | float64 | Real dimensionless cell-mean relative dielectric contrast chi=(epsilon_r−4)/4; axes batch,y-cell,x-cell |

Each output element describes one square cell of side0.8/30 m. The first row is the lowest y and the first column the lowest x; cell centers are −0.4+(j+0.5)*0.8/30 for zero-based index j. Relative permittivity is4*(1+chi). Submit the native30² image, without display upsampling, spatial cropping, fitted scale, rotation, translation, phase alignment or complex-valued surrogate contrast. The evaluator retains signed values, including ringing, without clipping. Evaluation requires uncentered cosine similarity and dynamic-range-normalized RMS error on the full spatial array, together with a physical scattered-field consistency check recomputed from the submitted real contrast. No global gain or phase is fitted during scoring.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 9, 10) | complex128 | Normalized scalar scattered electric field E_s/E0, dimensionless; axes batch,receiver,illumination; real/imaginary quadratures retained |
| illumination_angles | (1, 10) | float64 | Plane-wave propagation angle in rad, counterclockwise from positive x |
| receiver_angles | (1, 9) | float64 | Receiver polar angle in rad on the stated observation circle, counterclockwise from positive x |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation origin | existing author simulation | dimensionless | Reused deposited complex fields rather than a new generated acquisition |
| Frequency | 300000000 | Hz | Single-frequency illumination |
| Background relative permittivity | 4.0 | dimensionless | Known lossless homogeneous medium |
| Background conductivity | 0.0 | S/m | Perfect dielectric approximation |
| Background wavenumber | 12.575070131717595+0i | m^-1 | Deposited numerical calibration; preserve this stored value |
| Background wavelength | 0.4996540966663685 | m | Deposited wavelength convention |
| Incident amplitude | 1.0 | normalized electric-field unit | Phase-calibrated plane waves |
| Time convention | exp(+i omega t) | dimensionless | Outgoing Green function uses Hankel second kind |
| Image region | x and y each[−0.4,+0.4] | m | Finite square support in an unbounded background |
| Output grid | 30 y-cells by30 x-cells | cell | Physical cell means, no artificial display resolution |
| Output cell pitch | 0.8/30 | m | Equal Cartesian cell widths |
| Cell coordinates | half-cell offsets, x and y increasing with array index | m | Row-major flattening, y-cell then x-cell |
| Receiver radius | 1.0 | m | Circular acquisition geometry centered at the origin |
| Receiver count | 9 | receiver | All deposited receiver samples retained |
| Illumination count | 10 | source | All deposited plane-wave experiments retained |
| Angular sampling | np.arange(0,2*pi,2*pi/count) | rad | Stored angle arrays retain original floating-point values |
| Noise level | 1.0 | percent amplitude | Author metadata: random-phase phasor with magnitude1% of each noiseless complex sample, rather than Gaussian component variance |
| Spatial boundary | unbounded homogeneous outgoing-wave medium | dimensionless | No periodic physical boundary or reflecting image border |
| Pulse discretization | equal-area circular cell integration | dimensionless | Cell area dx*dy, equivalent disk radius sqrt(dx*dy/pi), analytic self term |

All angles and observations are supplied directly. Instrument geometry is independent of the unknown target. The existing noise realization is retained; no simulator seed or object-defining information is supplied.

## Method Hints

A Born iterative method alternates a regularized linear contrast solve using the current total field with a new physical field solve for the updated real permittivity. For a fixed field, the measurement matrix contains receiver Green coefficients multiplied by the local field. Tikhonov regularization stabilizes its poorly constrained spatial directions; the quadratic objective has the form ||K chi−d||²+alpha||chi||², where K is that current-field matrix, d stacks measured receiver/source data, and alpha is positive.

The internal state equation is a volume-integral equation. On a uniform grid its interaction can be evaluated by zero-extended FFT convolution, with a correctly integrated self term and a consistent complex adjoint. Distinguish a complex linear intermediate contrast from the real dielectric field used for physical propagation; a small complex-intermediate residual alone does not establish physical observation agreement.

## References

Wang and Chew, *An iterative solution of the two-dimensional electromagnetic inverse scattering problem*, International Journal of Imaging Systems and Technology1,100–108(1989). [Paper](https://doi.org/10.1002/ima.1850010111).

Batista, Adriano and Batista, *EISPY2D: An Open-Source Python Library for the Development and Comparison of Algorithms in Two-Dimensional Electromagnetic Inverse Scattering Problems*. [Author preprint, version2,2022](https://arxiv.org/abs/2111.02185v2); [related IEEE Access publication,2025](https://doi.org/10.1109/ACCESS.2025.3573679). The selected modern author BIM implementation and existing simulation data are distributed under GPL3. This is a local extraction of that modern implementation, not the1989 authors’ original software or a reproduction of all historical stochastic experiments.
