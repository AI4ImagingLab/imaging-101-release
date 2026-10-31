# Two-Planet Orbits from Stellar Radial Velocities

> Recover two planetary Keplerian orbits and instrument offsets and noise from irregular stellar Doppler measurements whose weak, overlapping signals span almost two decades.
> Domain: astronomy | Keywords: orbital inference, Doppler spectroscopy, Bayesian estimation | Difficulty: Hard

## Background

A planet's gravity causes its host star to move about their common centre of mass. The component of this motion along the line of sight shifts stellar absorption lines through the Doppler effect. Repeated spectroscopy can therefore detect companions too faint to image directly and measure their orbital periods, eccentricities and velocity amplitudes.

Long observing baselines reveal slowly orbiting planets, while dense observations help separate weaker short-period signals. Measurements from different spectrometers have independent velocity zero points and additional stellar or instrumental scatter. Irregular sampling, eccentric motion and overlapping periods make a multi-planet solution substantially more demanding than fitting a single sinusoid. Radial velocities constrain projected motion; they do not determine the orbital inclination or the true planet mass by themselves.

## Problem Description

Fit the full 401 radial velocities of HD 164922 with two noninteracting Keplerian companions. This benchmark uses a fixed two-planet interpretation of these historical observations, with no acceleration or curvature term. It does not ask whether additional planets exist in later observations. The measured radial velocity at time $t_n$ from instrument $s_n$ is

```math
y_n=\gamma_{s_n}+\sum_{p=1}^{2}K_p[\cos(\nu_p(t_n)+\omega_p)+e_p\cos\omega_p]+\epsilon_n,
\qquad \epsilon_n\sim\mathcal N(0,\sigma_n^2+j_{s_n}^2).
```

Here $y_n$ is the observed velocity, $\sigma_n$ its reported instrumental uncertainty, $\gamma_s$ an instrument zero point, $j_s$ its additional white-noise standard deviation, $K_p$ a positive velocity semi-amplitude, $e_p$ eccentricity, $\omega_p$ the argument of periastron in the stellar radial-velocity convention, and $\nu_p$ true anomaly. The period $P_p$ and inferior-conjunction epoch $T_p$ determine the phase through

```math
f_p=\pi/2-\omega_p,\quad
E_{c,p}=2\arctan[\tan(f_p/2)\sqrt{(1-e_p)/(1+e_p)}],\quad
t_{p}=T_p-\frac{P_p}{2\pi}(E_{c,p}-e_p\sin E_{c,p}),
```

```math
M_p(t)=2\pi\operatorname{frac}[(t-t_p)/P_p],\quad
E_p-e_p\sin E_p=M_p,\quad
\nu_p=2\arctan[\sqrt{(1+e_p)/(1-e_p)}\tan(E_p/2)].
```

The symbol $t_p$ denotes a periastron epoch, $E_{c,p}$ the eccentric anomaly at conjunction, $M_p$ the mean anomaly, $E_p$ the eccentric anomaly, and $\operatorname{frac}(a)=a-\lfloor a\rfloor$. Positive velocity denotes redshift along a line of sight pointing away from the observer; retain the signed measurements in m/s without range normalisation. The inclination remains unobserved. Instrument offsets and jitter must be inferred jointly with the orbital parameters.

Save **`output/reconstruction.npz`** containing all three real `float64` arrays:

| Key | Required shape | Units and order |
|---|---|---|
| reconstruction | (1, 2, 5) | Rows: long-period planet, short-period planet. Columns: period in days, inferior-conjunction BJD in days, dimensionless eccentricity, argument of periastron in radians, positive semi-amplitude in m/s. |
| instrument_offset | (1, 3) | Signed velocity zero points in m/s; instrument order 0, 1, 2. |
| instrument_jitter | (1, 3) | Positive additional white-noise standard deviations in m/s; instrument order 0, 1, 2. |

All arrays must be finite. Use $0\le e\le0.99$ and positive periods. Evaluation measures the maximum absolute error of each physical quantity across the corresponding rows or instruments, together with the root mean square residual divided by $\sqrt{\sigma_n^2+j_{s_n}^2}$. Conjunction epochs are compared modulo the reference period and arguments of periastron modulo $2\pi$; neither a velocity rescaling nor a fitted offset alignment is applied. These are checks against a validated standard point estimate, rather than independently known true orbits. The orbit rows cannot be exchanged.

## Data Description

`data/raw_data.npz` provides all observations, without nightly binning, selected subsets or target-derived calibration.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| time | (1, 401) | float64 | Observation times, full barycentric Julian days (days). |
| velocity | (1, 401) | float64 | Stellar radial velocities in the supplied instrumental convention (m/s). |
| uncertainty | (1, 401) | float64 | Positive reported instrumental velocity standard deviations (m/s). |
| instrument | (1, 401) | int64 | Instrument configuration index 0, 1 or 2 (dimensionless). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Target | HD 164922 | none | Star observed spectroscopically. |
| Instrument 0 | HIRES, pre-upgrade; 52 observations | none | Native instrument label k. |
| Instrument 1 | HIRES, post-upgrade; 276 observations | none | Native instrument label j. |
| Instrument 2 | APF; 73 observations | none | Native instrument label a. |
| Earliest and latest time | 2450275.9700771; 2457292.6796628 | BJD days | Full supplied time extent; zero epoch subtracted. |
| Reported uncertainty range | 0.798407554626 to 2.96615958214 | m/s | Per-observation standard deviations are supplied in the array. |
| Gravitating companions in selected model | 2 | count | Independent Keplerian reflex motions; mutual perturbations neglected. |
| Acceleration and curvature | 0; 0 | m/s/day; m/s/day squared | Fixed model terms; no fitted long-term trend. |
| Additional-noise model | Independent Gaussian, instrument-specific jitter | none | Jitter is unknown and added in quadrature to the reported uncertainty. |
| Position and velocity convention | Stellar line-of-sight reflex motion | none | Inferior conjunction uses true anomaly pi/2 minus the stellar argument of periastron. |

## Method Hints

Use well-converged maximum a posteriori estimation of the Gaussian heteroscedastic likelihood. In a stable eccentricity basis, $h_p=\sqrt{e_p}\cos\omega_p$, $q_p=\sqrt{e_p}\sin\omega_p$ and $a_p=\log K_p$, so $e_p=h_p^2+q_p^2$ and $\omega_p=\operatorname{atan2}(q_p,h_p)$. The negative log posterior is

```math
\mathcal L=\frac12\sum_n\left[\frac{(y_n-v_n)^2}{\sigma_n^2+j_{s_n}^2}+\log(2\pi(\sigma_n^2+j_{s_n}^2))\right]-\log\pi(\theta),
```

where $v_n$ is the full predicted radial velocity, $\theta$ contains all sixteen free parameters and $\pi$ denotes the following specified prior density. Period priors are proportional to $1/P$: long-period 500 to 5000 days and short-period 1 to 500 days. Conjunction-epoch priors are Gaussian with centres 2456779.0 and 2456277.6 BJD and standard deviations 300 and 25 days. Each $h$ and $q$ lies strictly between -1 and 1, each $a$ between $\log(0.001)$ and $\log(30)$, and each eccentricity lies between 0 and 0.99. Offsets have uniform priors strictly between -10 and 10 m/s; jitter has uniform priors strictly between 0 and 10 m/s. The prior is uniform in the stated eccentricity basis, without an extra Jacobian factor. These constraints define the selected historical analysis and the parameter labels.

A derivative-free direction-set search with explicit Kepler solves suits this low-dimensional problem. Scale the numerical coordinates so that large Julian-day epochs do not compromise convergence, tighten the objective stopping tolerance, and compare independent starts. The selected standard result includes precision continuation of the same physical posterior; reproducing an arbitrary early stopping point is not required. Keep the likelihood's variance-normalisation term: omitting it allows jitter to inflate without penalty. Reasonable starting periods are approximately 1206 and 75.8 days, with small eccentricities; the specified epoch priors identify the intended phase branch. No posterior convergence claim follows from a point estimate alone.

## References

- Fulton, Petigura, Blunt and Sinukoff (2018), *RadVel: The Radial Velocity Modeling Toolkit*, PASP 130, 044504. [Published method](https://doi.org/10.1088/1538-3873/aaaaa8), [author manuscript](https://arxiv.org/abs/1801.01947). The selected method is the maximum a posteriori fit illustrated for this system, using the complete two-planet analysis.
- Fulton et al. (2016), *Three Temperate Neptunes Orbiting Nearby Stars*. [Observation paper](https://arxiv.org/abs/1607.00007). The supplied historical radial velocities constrain the selected two-companion model.
