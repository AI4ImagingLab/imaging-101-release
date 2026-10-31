# Colloidal Particle Geometry from Flow Holography

> Recover the three-dimensional centers, radii and refractive indices of two suspended colloidal spheres from a real microscopy intensity frame despite missing optical phase and coupled scattering parameters.
> Domain: mechanics | Keywords: colloidal flow imaging, inverse scattering, particle metrology | Difficulty: Hard

## Background

Suspended particles scatter coherent light into concentric interference patterns. Their fringe structure encodes distance from the focal plane, while angular scattering and contrast encode particle size and optical composition. Recovering these properties supports colloidal suspension metrology and flow cytometry without requiring particles to sit on a surface.

A single camera frame records intensity rather than complex electric field. Holographic particle metrology combines the incident light with the particle's scattered field to recover depth information otherwise absent from an ordinary brightfield image. Finite fields, nonuniform illumination, other particles and camera noise limit the accuracy of an isolated-sphere description.

## Problem Description

The frame contains two homogeneous, nonabsorbing spheres suspended in water. Recover one physical parameter vector for each specified region of interest. Rows of the output follow the ROI order below. The incident wave is polarized along positive x and propagates along negative z; particles above the focal plane have positive z. Pixel coordinates are zero-based x=column and y=row; the measurement plane is z=0.

```math
J(u,v)=\frac{I(u,v)-d}{B(u,v)-d},\qquad
H(u,v;q)=|\hat{\mathbf{x}}+\mathbf E_s(u,v;q)|^2.
```

I is the recorded camera count, B an independently acquired particle-free background and d the dark count. Es is the dimensionless electric field scattered by a homogeneous sphere, with its incident phase included. Use the vector Lorenz-Mie solution to Maxwell's equations, with vacuum wavelength lambda, medium index nm, particle radius a and real index np. Its sphere coefficients are

```math
a_l=\frac{m\psi_l(mx)\psi'_l(x)-\psi_l(x)\psi'_l(mx)}{m\psi_l(mx)\xi'_l(x)-\xi_l(x)\psi'_l(mx)},\quad
b_l=\frac{\psi_l(mx)\psi'_l(x)-m\psi_l(x)\psi'_l(mx)}{\psi_l(mx)\xi'_l(x)-m\xi_l(x)\psi'_l(mx)}.
```

Here l is positive multipole order, x=2 pi nm a/lambda is size parameter, m=np/nm is relative refractive index, psi is the regular Riccati-Bessel function, xi the outgoing Riccati-Hankel function and primes denote derivatives. Combine these coefficients with outgoing vector spherical waves at the particle-to-pixel displacement and the incident phase exp(-i k z), where k=2 pi nm/lambda in inverse micrometres and z is particle height in micrometres. Retain sufficient multipoles for numerical convergence. All three scattered electric-field components contribute to intensity.

To make the target reproducible under real model mismatch, each ROI defines an isolated-sphere conditional fit. The fitting pixels are every pixel except those attaining that ROI's maximum normalized intensity, following the observed-maximum exclusion convention. Do not use neighboring-particle fitted information. The required conditional objective for each particle is

```math
L(q)=\frac1N\sum_{p\in V}\log\left(1+\left[\frac{H(p;q)-J(p)}{0.05}\right]^2\right).
```

V is the declared valid-pixel set and N its size. The dimensionless Cauchy residual scale 0.05 is a robust modeling scale, not an independently calibrated photon-noise standard deviation. This objective downweights dust and neighboring-particle mismatch. Multiple parameter vectors may explain a hologram; evaluation accepts any vector within the physical bounds with sufficiently low objective. No agreement with a hidden parameter vector or particular sequence of iterations is required.

Save `output/reconstruction.npz` containing `reconstruction`, real float32 or float64, shape `(1, 2, 5)`, axes batch, particle, parameter. Parameter order is `(x,y,z,a,np)`: x/y/z in image pixels, radius a in micrometres, and np dimensionless. x/y must lie within the particle's ROI, z is between100 and350pixels, a between0.5 and2micrometres and np between1.35 and1.8. The absorption index is fixed at zero. Each particle's objective must be at most1.342 and1.161, respectively. Every parameter enters the forward check. No translation, rescaling, reflection, cropping of the specified ROI or fitted output alignment is used. Passing establishes a good conditional model fit; it does not establish independent particle truth.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| image | (1,1024,1280) | uint8 | Raw camera intensity, counts; batch,row,column |
| background | (1,1024,1280) | uint8 | Independently recorded particle-free illumination, counts; batch,row,column |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Vacuum wavelength | 0.447 | micrometre | Coherent illumination |
| Pixel spacing | 0.048 | micrometre/pixel | Object-space magnification |
| Medium index | 1.34 | 1 | Water suspension |
| Dark count | 13 | count | Nominal author-provided offset, subtracted from frame and background; no independent dark frame is supplied |
| ROI1 | x=809,y=187,width=228,height=228 | pixel | Upper-right particle; half-open pixel range |
| ROI2 | x=222,y=635,width=200,height=200 | pixel | Lower-left particle; half-open pixel range |
| Polarization | x | 1 | Incident electric field direction |
| Propagation | negative z | 1 | Camera plane z=0; particle z positive |
| Sphere absorption index | 0 | 1 | Nonabsorbing homogeneous sphere |

The complete source frame and background retain their native uint8 precision. ROIs are measurement windows fixed from observed fringes; no inferred radius, index or height is supplied as calibration.

## Method Hints

Use a vector electromagnetic scattering model and a nonlinear inverse fit. Coarse global exploration of depth, radius and refractive index can avoid fringe-order minima; local refinement can recover subpixel centers. Robust residuals reduce the influence of dust and neighboring fringes. Compare candidate solutions by the declared objective rather than assuming a unique particle parameter vector.

## References

S.-H. Lee et al., Characterizing and tracking single colloidal particles with video holographic microscopy, Optics Express15,18275–18282(2007), doi:10.1364/OE.15.018275.

F.C.Cheong et al., Flow visualization and flow cytometry with holographic video microscopy, Optics Express17,13071–13079(2009), doi:10.1364/OE.17.013071.

C.Martin et al., In-line holographic microscopy with model-based analysis, Nature Reviews Methods Primers2,83(2022).
