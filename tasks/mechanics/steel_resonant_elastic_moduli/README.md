# Steel Resonant Elastic Moduli

> Infer the compressional and shear stiffnesses of a free steel cylinder from a partially observed elastic resonance spectrum with unknown mode correspondences.
> Domain: mechanics | Keywords: resonant ultrasound, inverse elasticity, spectral inference | Difficulty: Hard

## Background

A solid body supports mechanical resonances determined by its stiffness, mass distribution and shape. A pair of weakly contacting piezoelectric transducers can excite and detect these vibrations without destroying the sample. The frequencies constrain elastic properties throughout the body's volume, rather than only at a surface contact.

Resonant ultrasound is useful for material characterization, manufacturing assessment and studies of elastic changes with temperature. Recovering stiffness from many resonances requires a continuum elastic model. Missing modes and frequency degeneracies complicate correspondence between detected peaks and model eigenmodes.

## Problem Description

Recover the two independent isotropic elastic moduli of the measured commercial 4140 steel cylinder. Let c11 be longitudinal stiffness and c44 shear modulus, both in GPa. Define Lamé coefficients lambda=c11−2c44 and mu=c44. For indices i,j,k,l in {x,y,z}, the stiffness tensor is C(i,j,k,l)=lambda delta(i,j)delta(k,l)+mu[delta(i,k)delta(j,l)+delta(i,l)delta(j,k)], where delta is the Kronecker delta.

The vibration displacement u(r) satisfies the traction-free elastic eigenproblem

```math
-\nabla\cdot\bigl(C:\varepsilon(u)\bigr)=\rho\omega^2u,
\qquad (C:\varepsilon(u))n=0,
\qquad \varepsilon(u)=\tfrac12(\nabla u+\nabla u^T).
```

Here r is position, rho is homogeneous density, omega angular frequency, epsilon the strain tensor, and n the outward surface normal. The colon denotes contraction over the two strain indices. Ordinary frequency is omega/(2pi). Rigid translation and rotation produce six zero-frequency modes. The data do not provide model mode indices. Match measured positive frequencies to the calculated sorted spectrum while allowing undetected modes. Exact duplicate frequencies in this processed author list represent assigned degeneracy multiplicities; they are not independent repeated sensor measurements. The correspondence still has to be recovered.

Save **output/reconstruction.npz** with these real float64 arrays:

| Output key | Shape | Units and axis order |
|---|---|---|
| reconstruction | (1, 2) | GPa; batch, [c11, c44] |
| resonances | (1, 76) | MHz; batch, ascending nonrigid eigenmode frequency |

Compute the reported spectrum using the total-degree-14 Cartesian polynomial Rayleigh–Ritz discretization: each displacement component is expanded in x^p y^q z^r, with nonnegative integer exponents p+q+r<=14, in the centered cylinder. Integrate volume moments analytically; remove the six lowest rigid modes and retain the next 76. No arbitrary phase, scale, frequency alignment, target rescaling or cropping is used. Both moduli and the entire theoretical spectrum are evaluated by absolute physical-unit errors. Also evaluate measured-frequency consistency using 100 sqrt(sum(w(fobs/fcalc−1)^2)/number of nonzero w), in percent, after monotone matching. w denotes the supplied author analysis preference, and fobs/fcalc the paired measured/calculated frequencies. Method agreement does not establish an independently measured true elastic tensor.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frequencies | (1, 70) | float64 | Positive measured resonances in MHz, ascending; author degeneracy repetitions retained |
| analysis_weights | (1, 70) | float64 | Dimensionless author analysis preferences (0, 0.5 or 1), not calibrated measurement precision |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Material | Commercial 4140 steel | material | Homogeneous isotropic continuum approximation |
| Shape | Solid circular cylinder | geometry | Cylinder axis is z |
| x diameter | 1.900094 | cm | Full diameter, not radius |
| y diameter | 1.900094 | cm | Full diameter, not radius |
| z length | 1.338456 | cm | Full height |
| Mass | 29.7024 | g | Density is mass divided by pi*diameter_x*diameter_y*height/4 |
| Coordinates | Centered Cartesian x,y,z | cm | All three coordinates extend symmetrically about zero |
| Boundary | Free traction | mechanical condition | Weak transducer contact approximates an unconstrained body |
| Excitation/readout | Piezoelectric drive and pickup, lock-in detection | instrument | Complete body resonances inferred from scanned ultrasonic response |
| Frequency calibration | Supplied peak frequencies | MHz | No absolute timing correction or unknown frequency scale is fitted |
| Noise | No calibrated variance supplied | MHz | Peak identification, shape error and isotropic-model discrepancy affect residuals |
| Degenerate observations | Author-processed multiplicities retained | convention | Exact repeated values share the same measured peak; no independent-noise claim |
| Analysis preferences | Supplied vector | dimensionless | Retain zero-preference observations; they do not contribute to the weighted fit metric |

## Method Hints

Use a continuum Rayleigh–Ritz inverse eigenvalue method. Analytic volume moments yield mass and stiffness matrices. Mirror symmetries separate the calculation into eight independent blocks. A damped Gauss–Newton method can adjust the two moduli using finite-difference sensitivities. The correspondence step can be implemented as monotone assignment between observed and calculated frequencies, allowing skipped model modes. Low-frequency modes offer a useful initial estimate before fitting the complete spectrum. Assess mode assignments and physical residuals alongside the fitted constants.

## References

1. Torres, Flores-Betancourt and Hermann. *RUScal: Software for the analysis of resonant ultrasound spectroscopy measurements*. Journal of the Acoustical Society of America 151, 3547–3563 (2022). [DOI:10.1121/10.0011397](https://doi.org/10.1121/10.0011397).
2. Balakirev et al. *Resonant ultrasound spectroscopy: The essential toolbox*. Review of Scientific Instruments 90, 121401 (2019). [DOI:10.1063/1.5123165](https://doi.org/10.1063/1.5123165).
