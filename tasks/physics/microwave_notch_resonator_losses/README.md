# Microwave Notch Resonator Losses

> Recover internal and coupling dissipation and a complex microwave resonance from a measured transmission sweep distorted by cable delay and instrumental gain.
> Domain: physics | Keywords: microwave spectroscopy, dissipation inference, complex scattering | Difficulty: Hard

## Background

Microwave resonators store electromagnetic energy in a narrow frequency band. Energy escapes through the coupling to a transmission line and through intrinsic dissipative processes. These losses determine the linewidth and help characterize superconducting circuits, dielectric materials and sensitive microwave detectors.

A vector network analyzer records both quadratures of the transmitted field. The resulting complex resonance carries more information than its amplitude alone, but propagation through cables and imperfect impedance matching rotate and deform its trajectory. Separating intrinsic dissipation from coupling therefore requires a consistent complex scattering model.

## Problem Description

The supplied complete real sweep measures a notch resonator in transmission. At frequency f in Hz, the dimensionless complex scattering coefficient is

```math
S(f)=a\exp(i\alpha-2\pi i f\tau)
\left[1-\frac{Q_l\exp(i\phi)}{|Q_c|\{1+2iQ_l(f-f_r)/f_r\}}\right]+\epsilon(f).
```

Here fr is resonance frequency, Ql is loaded quality factor, |Qc| is the magnitude of complex coupling quality, phi is mismatch phase in radians, a is positive dimensionless gain, alpha is instrumental phase in radians, tau is cable delay in seconds, and epsilon is measurement/model error. The complex coupling convention is Qc_complex=|Qc| exp(-i phi). Diameter-corrected real coupling is Qc=|Qc|/cos(phi), and intrinsic quality Qi satisfies 1/Qi=1/Ql-1/Qc. Gain, phase, mismatch and delay are unknown and must be estimated from the observations. The finite bandwidth and correlated calibration uncertainty complicate separation of these effects. The source instrument file contains no established independent truth for the losses.

Save `output/reconstruction.npz` relative to the task workspace with all these arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| resonance_frequency | (1, 1) | float64 | Resonance frequency fr in Hz, axes batch,scalar |
| quality_factors | (1, 3) | float64 | Dimensionless real quality factors, ordered Ql,Qc,Qi with diameter correction |
| reconstruction | (1, 1001) | complex128 | Fitted raw-scale complex S21, dimensionless voltage ratio, axes batch,frequency |

Evaluation requires agreement in every physical output and the full complex response with a validated standard-method reference. Frequency uses relative absolute error; quality factors use maximum componentwise relative absolute error; complex response uses relative L2 error. Observation consistency uses norm(Sfit-Smeasured)/norm(Smeasured). There is no phase, gain or delay alignment during scoring. The full response must include estimated instrumental effects. Method agreement measures reproducibility, not independent accuracy of physical loss values.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frequency | (1, 1001) | float64 | Increasing native frequency samples, Hz; axes batch,frequency |
| scattering | (1, 1001) | complex128 | Measured raw S21, dimensionless complex voltage ratio; axes batch,frequency |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition timestamp | 2012-08-08 15:06:23 | local clock, zone unspecified | Native instrument record |
| Sample label | NIST | identifier | Author instrument filename |
| Nominal temperature | 275 | mK | Native instrument filename |
| Nominal power | -90 | dBm | Native instrument filename; no independent sample-plane power calibration supplied |
| Reference impedance | 50 | ohm | Touchstone instrument convention |
| Native frequency unit | GHz converted by 1e9 | Hz/GHz | No resampling or cropping |
| Frequency range | 5922000000 to 5923000000 | Hz | Complete supplied sweep |
| Frequency increment | 1000 | Hz | Uniform native sampling |
| Complex convention | real plus i imaginary | dimensionless | Touchstone S RI format |
| Model boundary | one isolated notch, constant gain, linear cable phase | - | No baseline slope correction; finite measured band |
| Noise | not independently calibrated | - | Residuals include instrumental and model discrepancy; no Gaussian variance claim |

## Method Hints

An algebraic complex-circle fit combined with resonance phase fitting can separate the cable delay, off-resonance gain and dissipative parameters. Correcting the circle diameter for mismatch phase prevents coupling losses from being silently assigned to internal dissipation. Fit complex quadratures consistently and inspect both raw and normalized residuals.


The evaluated standard recovery uses the constrained algebraic circle solution without subsequent nonlinear geometric or radial-weighted circle refinement. For complex samples after removal of the trial cable phase, subtract their complex mean and write each centered point as (x(j),y(j)). For real coefficient vector a=(a0,a1,a2,a3), define

```math
\underset{a}{\operatorname{minimize}}\;
\sum_j\left[a_0(x(j)^2+y(j)^2)+a_1x(j)+a_2y(j)+a_3\right]^2,
\qquad a_1^2+a_2^2-4a_0a_3=1.
```

Equivalently let v(j)=(x(j)^2+y(j)^2,x(j),y(j),1) be a column vector, let M be the sum of its outer products v(j)v(j)-transpose, and let C have C(0,3)=C(3,0)=-2, C(1,1)=C(2,2)=1, with all other entries zero and indices0,...,3. Select the smallest nonnegative root eta of det(M-eta C)=0 and its null vector, using the root branch reached from zero. The centered circle has center (-a1/(2a0),-a2/(2a0)) and radius sqrt(a1²+a2²-4a0a3)/(2 abs(a0)); add the removed mean back to its center. Apply this same circle convention in cable/gain calibration and the final normalized resonance fit. Then fit the centered resonance phase and use the diameter/mismatch corrections already specified above. The gain is constant and the amplitude-slope correction is zero. Additional nonlinear circle refinement or a full complex nonlinear refit defines a different estimator, even when it slightly improves the observed residual. This distinction matters for the standard-method comparison; it is not a claim that the algebraic result is independent physical truth.

The cable delay minimizes the unweighted sum of squared radial deviations from that algebraically fitted circle after multiplication by exp(2pi i f tau). The centered resonance phase fit minimizes squared circular angular residuals to theta0+2 atan(2 Ql(1-f/fr)), using all sweep points with equal weights. In the delay-corrected raw circle, the off-resonance point is its center plus its radius times exp(i(theta0+pi)); its modulus and argument give gain a and phase alpha. Normalize by this complex gain before the final circle fit. In that normalized circle, with center (xc,yc) and radius r, set phi=-asin(yc/r), abs(Qc)=Ql/(2r), and use the diameter corrections already stated. These conditional geometric criteria, rather than a joint complex-response least-squares alternative, select the evaluated standard estimate.

## References


S. Probst, F. B. Song, P. A. Bushev, A. V. Ustinov and M. Weides. Efficient and robust analysis of complex scattering data under noise in microwave resonators. Review of Scientific Instruments 86, 024706 (2015). [DOI:10.1063/1.4907935](https://doi.org/10.1063/1.4907935); [arXiv:1410.3365](https://arxiv.org/abs/1410.3365).
