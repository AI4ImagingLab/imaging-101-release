# Solar Photospheric Vector Mapping from Stokes Spectra

> Reconstruct nine spatial maps of the solar photosphere from resolved polarized iron-line spectra whose magnetic, velocity and thermodynamic signatures overlap.
> Domain: astronomy | Keywords: spectropolarimetric imaging, polarized radiative transfer, magnetic mapping | Difficulty: Hard

## Background

Magnetic fields split atomic energy levels and produce polarized spectral lines. The strength, orientation and wavelength dependence of the four Stokes components encode the magnetic field and the motion of the emitting atmosphere. A spatially resolved spectropolarimeter measures a spectrum at every image position, allowing maps of photospheric conditions to be inferred without visiting the Sun.

Such maps help study sunspots, magnetic energy storage and the coupling between solar convection and magnetism. The polarization signals are weaker than the intensity, several atmospheric parameters have similar effects on line shape, and finite instrument transmission broadens the spectra. The inferred atmosphere therefore depends on both physical assumptions and measurement sensitivity.

## Problem Description

The observations are a contiguous 64 by 96 pixel region of an SST/CRISP sunspot observation on 2019-05-10. At every position infer a depth-independent magnetic field and line parameters in a Milne–Eddington atmosphere. The source function varies linearly with continuum optical depth: S(tau)=S0+S1*tau. The emergent Stokes vector is


```math
\boldsymbol{s}(\lambda)=S_0\boldsymbol{e}_I+
\mu S_1\boldsymbol{K}(\lambda)^{-1}\boldsymbol{e}_I,
\qquad \boldsymbol{e}_I=(1,0,0,0)^\mathsf{T}.
```


Here s=(I,Q,U,V) is normalized to the continuum intensity, mu is the viewing-angle cosine, and K is the dimensionless polarized propagation matrix,


```math
K=\begin{pmatrix}
\eta_I&\eta_Q&\eta_U&\eta_V\\
\eta_Q&\eta_I&\rho_V&-\rho_U\\
\eta_U&-\rho_V&\eta_I&\rho_Q\\
\eta_V&\rho_U&-\rho_Q&\eta_I
\end{pmatrix}.
```


The eta coefficients describe absorption and dichroism, while rho describes magneto-optical dispersion. They combine the pi and two sigma Zeeman component profiles of both iron lines. For each transition, the wavelength shift is the sum of lambda0*v/c and the Zeeman shift proportional to B*lambda0^2*(g_upper*M_upper-g_lower*M_lower). Lambda0 is the rest wavelength, v the line-of-sight velocity, c the speed of light, B field strength, g the Landé factor and M the magnetic quantum number. Voigt absorption and Faraday–Voigt dispersion use the common Doppler width, damping and line-to-continuum opacity ratio. Line strengths are normalized separately within each polarization group.

The detector measures each Stokes spectrum after convolution with the supplied spectral transmission and sampling at the observed wavelengths. There is no spatial deconvolution in this acquisition model. The inversion is nonlinear and nonunique: thermodynamic parameters trade off, a single line pair has limited depth information, and azimuth has an intrinsic pi ambiguity.

Save `output/reconstruction.npz` relative to the task workspace, containing:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 64, 96, 9) | float64 | Real physical parameter maps; axes batch, image row, image column, parameter, in the exact order below. |

| Parameter index | Quantity | Unit and convention |
|---|---|---|
| 0 | Magnetic field strength B | G; nonnegative |
| 1 | Field inclination | rad; [0,pi], measured from the line of sight |
| 2 | Field azimuth | rad; [0,pi], projected-field angle in the measured Q/U reference frame |
| 3 | Calibrated line-of-sight velocity | km/s; positive is redshift; synthesis velocity plus cavity_velocity plus 0.45 km/s |
| 4 | Doppler width | angstrom; wavelength width, shared by both lines |
| 5 | Line opacity ratio | dimensionless; relative to continuum, first iron line normalization |
| 6 | Voigt damping | dimensionless |
| 7 | Source constant S0 | continuum intensity units |
| 8 | Source gradient S1 | continuum intensity units per dimensionless optical depth |

All image pixels and all nine parameters enter an independent full Stokes synthesis. Each of I,Q,U,V must meet its whitened RMS boundary: respectively6.034479885252131,13.124827389436739,14.737955362158246,14.723503203784697, using the acquisition fitting scales below. These unchanged observation-quality limits account for measured-model mismatch and do not claim unit-noise accuracy. There is no parameter-map agreement to a particular local fit, spatial alignment, fitted gain or parameter permutation. Equivalent spectra within the declared priors are accepted.

The admissible conditional atmosphere has the following box prior in the quantity order above: lower=(0,0,0,-13,0.001,0.01,1e-5,0,-10), upper=(6000,pi,pi,13,0.2,120,10,10,10), with each component in its declared units. The velocity bound applies before adding the public cavity_velocity and0.45km/s zero-point correction; all other bounds apply to the reported quantities directly. The azimuth representative is [0,pi]. Domain boundaries have1e-10 numerical slack. The inverse target minimizes the unweighted sum of squared residuals after the four Stokes scales are applied, subject to this box; the evaluator accepts any atmosphere attaining all four stated observation-quality limits rather than enforcing a particular optimizer path.


## Data Description

Load the following observation and calibration arrays from `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| stokes | (1, 64, 96, 4, 30) | float64 | Observed I,Q,U,V spectra; dimensionless continuum-normalized intensity; spatial rows and columns retain detector order. |
| wavelength | (1, 30) | float64 | Absolute sampled wavelengths in angstrom, increasing within each line window. |
| cavity_velocity | (1, 64, 96) | float64 | Instrument cavity velocity correction in km/s, to add after fitting. |
| spectral_psf_6301 | (1, 75) | float64 | Instrument transmission samples, dimensionless and unit sum; ordered increasing wavelength offset at 0.005 angstrom spacing. |
| spectral_psf_6302 | (1, 25) | float64 | Instrument transmission samples, dimensionless and unit sum; ordered increasing wavelength offset at 0.015 angstrom spacing. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument and date | SST/CRISP, 2019-05-10 | calendar date | Resolved solar iron-line spectropolarimetry |
| Pixel spacing | 0.059 | arcsec | Equal image row and column spacing; native detector order |
| Viewing cosine mu | 0.93 | dimensionless | Angle between observer direction and local solar normal |
| Rest wavelengths | 6301.4995, 6302.4931 | angstrom | First and second Fe I line |
| First-line quantum numbers | Jlower=2, Jupper=2, glower=1.84, gupper=1.50 | dimensionless | Anomalous Zeeman pattern |
| Second-line quantum numbers | Jlower=1, Jupper=0, glower=2.49, gupper=0 | dimensionless | Three Zeeman components |
| Relative oscillator strengths | 1, 10^(-0.442) | dimensionless | Opacity scaling of first and second line |
| Spectral windows | First 17 samples; last 13 samples | sample index | Two independently degraded spectral regions |
| Fine spectral spacing | 0.005, 0.015 | angstrom | First and second region; include five fine samples beyond either observed edge |
| Fine-grid convention | Start at first observed wavelength minus five steps; count round(last offset/step)-round(first offset/step)+11 | dimensionless | Offsets are measured from 6302.4931 angstrom; nearest-grid sampling |
| PSF boundary | Constant continuation of each end value | dimensionless | Independent region boundaries; no spectral periodic wrap |
| Spectral line extent | Both lines evaluated across both complete fine regions | angstrom | Includes cross-line wings |
| Stokes fitting scales I,Q,U,V | 0.003, 0.0003, 0.0003, 0.003/3.5 | normalized intensity | Weighting convention; Q/U and V scales are enhanced fitting weights, not independently measured photon noise |
| Velocity zero correction | cavity_velocity + 0.45 | km/s | Add to synthesis velocity to obtain submitted calibrated velocity |
| Light speed | 299792.458 | km/s | Nonrelativistic Doppler convention |
| Zeeman constants | e=4.80320441e-10, electron mass=9.10938188e-28, c=2.99792458e10 | electrostatic cgs, g, cm/s | Wavelength splitting coefficient e/(4e8*pi*m*c^2) for wavelengths in angstrom and B in G |
| Source and atmosphere | Linear source; depth-independent propagation coefficients | dimensionless | Semi-infinite Milne–Eddington atmosphere |

## Method Hints

Analytic response functions provide the change of each Stokes sample with each atmospheric parameter. A bounded Levenberg–Marquardt fit can use these responses to resolve nonlinear parameter interactions while limiting steps in weakly constrained directions. Scale the physical parameters before assembling normal equations. A second fit initialized from a spatially smoothed first-pass atmosphere can stabilize spatial maps; this initialization does not alter the instrumental forward model. The continuum and polarization symmetries provide useful physical checks of a local implementation.

## References

- Orozco Suárez, D. and del Toro Iniesta, J. C. (2007). The usefulness of analytic response functions. Astronomy & Astrophysics 462, 1137–1145. [Published article](https://doi.org/10.1051/0004-6361:20066201); [author manuscript](https://arxiv.org/abs/1211.1502). Supports the analytic Milne–Eddington transfer and response functions.
- de la Cruz Rodríguez, J. (2019). A method for global inversion of multi-resolution solar data. Astronomy & Astrophysics 631, A153. [Published article](https://doi.org/10.1051/0004-6361/201936635). The present task selects the author's ordinary spatially resolved Milne–Eddington example and its two-pass initialization; it does not reproduce that article's multi-resolution spatially coupled experiment.
