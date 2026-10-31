# Dune Radar Stolt Migration

> Reconstruct a signed subsurface reflection image from a complete measured ground-penetrating radar profile despite diffraction, finite aperture, background arrivals and uncertain propagation geometry.
> Domain: earth science | Keywords: electromagnetic imaging, wave migration, subsurface imaging | Difficulty: Hard

## Background

Ground-penetrating radar transmits short electromagnetic pulses and records reflections from contrasts in dielectric properties. The returned signals can reveal sedimentary layers, buried objects, water-related boundaries and near-surface geological structure without excavation. A moving antenna produces a radargram whose horizontal coordinate follows the survey line and whose vertical coordinate is recorded travel time.

Reflectors need not lie directly beneath an antenna. An isolated object produces a diffraction hyperbola across neighbouring traces, and dipping layers appear displaced in an unmigrated radargram. Wave migration focuses these responses into an interpretable subsurface image, making radar useful in sedimentology, archaeology, infrastructure assessment and environmental surveys.

## Problem Description

The acquisition is a real common-offset GSSI radar survey across a dune. Use the locally planar, homogeneous-background, zero-offset approximation to reconstruct an initial reflected wavefield rather than absolute dielectric permittivity. Let $P(x,z,t)$ denote the exploding-reflector wavefield, $x$ along-profile distance in metres, $z$ nominal depth in metres, and $t$ corrected travel time in seconds. Its scalar model is

```math
\frac{\partial^2P}{\partial t^2}
=\left(\frac{v}{2}\right)^2\left(\frac{\partial^2P}{\partial x^2}
+\frac{\partial^2P}{\partial z^2}\right),
\qquad d(x,t+t_0)=P(x,0,t)+c(x)+b(t)+\epsilon(x,t).
```

Here $v$ is the nominal two-way electromagnetic propagation velocity, $t_0$ the instrument time delay, $d$ the measured signed receiver amplitude, $c(x)$ a trace-dependent constant offset, $b(t)$ a background common to traces, and $\epsilon$ noise/model error. The initial image is $I(x,z)=P(x,z,0)$. Pulse shape, instrument gain and finite bandwidth are retained in this image, so it is a signed reflection image in instrument amplitude units rather than a calibrated material-property map. Positive depth points below the nominal acquisition surface. The surveyed surface is sloping; the selected method uses a planar approximation, while the supplied independent elevation survey supports an explicitly identified post-migration elevation display.

Save `output/reconstruction.npz` relative to the task workspace:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 458, 1040) | float64 | Real signed migrated wavefield in instrument amplitude units (receiver counts); axes batch, nominal depth z, profile x. The depth origin is zero, z index j corresponds to j × 0.006575342465753425 m, and x follows position_m. Preserve physical amplitude scale and polarity. |

The output is the planar image before display-only terrain shifting. Its 458 corrected-time samples start exactly at the stated delay on a uniform grid with the acquired time interval. Evaluation uses cosine NCC without mean subtraction and NRMSE defined as RMS error divided by the comparison image's maximum minus minimum. No crop, translation, fitted scale, magnitude conversion or terrain-based alignment is applied. This real-data task measures reproducibility of a standard discrete migration result, not agreement with independently known subsurface truth.

## Data Description

`data/raw_data.npz` contains all recorded radar samples and independently measured survey geometry:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| radar | (1, 512, 1040) | float64 | Signed receiver amplitudes in instrument counts; axes batch, recorded time, profile trace. Original integer values are preserved exactly in float64. |
| time_ns | (1, 512) | float64 | Recorded time coordinates in ns, including the instrument delay. |
| position_m | (1, 1040) | float64 | Trace positions in m along the profile, with first trace at zero. |
| topo_position_m | (1, 36) | float64 | Along-profile positions in m of independently surveyed surface-elevation stations. |
| surface_elevation_m | (1, 36) | float64 | Surface elevation in m in the source survey's vertical datum, paired with topo_position_m. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Recorded time window | 0.0 to 48.0 | ns | 512 sampled times, both endpoints included in the supplied coordinates. |
| Sampling interval | 0.09393346379647749 | ns | Preserved float64 interval of the supplied time axis. |
| Trace density | 50.0 | traces/m | GSSI acquisition-header spatial calibration. |
| Trace spacing | 0.02 | m | Uniform physical spacing derived from trace density. |
| Complete trace count | 1040 | traces | Entire measured line, with x from 0 to 20.78 m. |
| Instrument time delay t0 | 5.0 | ns | Fixed nominal arrival-time calibration used for continuous time-zero resampling. |
| Background wave velocity v | 140000000.0 | m/s | Selected homogeneous-background model, equivalent to 0.14 m/ns; not a measured permittivity map. |
| Output depth spacing | 0.006575342465753425 | m | Half the nominal velocity times the time interval in seconds. |
| Coordinates | x along profile, z downward | dimensionless | Nominal planar imaging coordinates; elevation is upward in the independent survey datum. |
| Offset model | Zero-offset approximation | dimensionless | Treat transmitter and receiver as coincident for this scalar migration. |
| Noise calibration | Unavailable | counts | No independent sample-wise variance or noise-free radargram is supplied. |
| Boundary model | Finite aperture, periodic Fourier transform | dimensionless | Linear edge weighting precedes migration; temporal spectral interpolation uses the stated endpoint-extension convention. |

Surface elevation is independent acquisition geometry, not an estimate from the unknown subsurface. A vertical display shift using it does not correct wave propagation for topography or lateral velocity variation.

## Method Hints

Constant-velocity Stolt migration maps temporal frequency to vertical wavenumber using the exploding-reflector dispersion relation. For angular temporal frequency $\omega$ in rad/s and spatial wavenumbers $k_x,k_z$ in rad/m,

```math
\omega_* = \frac{v}{2}\sqrt{k_x^2+k_z^2},\qquad
\widehat I(k_x,k_z)=\frac{k_z}{\sqrt{k_x^2+k_z^2}}\,
\widehat d(k_x,\omega_*).
```

The hats denote discrete Fourier transforms of the time-corrected, background-removed field and migrated image. A continuous linear interpolation onto times $t_0+j\Delta t$ avoids nearest-sample rounding of the delay; $\Delta t$ is the acquired interval and $j$ a nonnegative sample index. Remove the per-trace mean and common mean trace before migration. A separable linear edge weight is $\min(1,i/100,(N_x-1-i)/100)\min(1,j/40,(N_t-1-j)/40)$, where $i,j$ index the 1040 traces and 458 corrected-time samples, and $N_x,N_t$ are those counts. This fixes the finite-aperture convention and output amplitude normalization.

Use linear interpolation in temporal frequency at fixed horizontal wavenumber; the selected standard implementation extends the last measured frequency value for queries above the sampled band, omits the final temporal Nyquist output bin, and sets the zero-frequency/zero-wavenumber coefficient to zero. Endpoint extension is a numerical boundary approximation, not a strictly bandlimited wave model. A zero-outside-band diagnostic is useful for checking its small but measurable effect. The obliquity factor above is dimensionless because the output depth sampling is tied to $z=vt/2$; retain the signed discrete FFT normalization.

## References

Lilien, D. A., Hills, B. H., Driscol, J., Jacobel, R., and Christianson, K. (2020). *ImpDAR: an open-source impulse radar processor*. Annals of Glaciology 61(81), 114–123. [doi:10.1017/aog.2020.44](https://doi.org/10.1017/aog.2020.44).

Plattner, A. (2020). *GPRPy: Open-source ground-penetrating radar processing and visualization software*. The Leading Edge 39(5), 332–337. [doi:10.1190/tle39050332.1](https://doi.org/10.1190/tle39050332.1). This task applies the former reference's migration implementation to the latter project's complete real dune acquisition; it does not claim an unchanged reproduction of either paper's complete processing figures.
