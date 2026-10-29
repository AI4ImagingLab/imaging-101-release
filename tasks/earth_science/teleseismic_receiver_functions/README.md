# Teleseismic Receiver Functions

> Reconstruct radial and transverse crustal impulse responses from a real three-component earthquake record whose long source wavelet and weak spectral bands obscure converted seismic arrivals.
> Domain: Earth science | Keywords: seismic imaging, wave conversion, regularized deconvolution | Difficulty: Medium

## Background

A seismic wave crossing a boundary between rocks with different elastic properties can change its propagation mode. A small portion of an incident compressional wave converts into shear motion beneath a recording station. The timing and polarity of these arrivals contain information about the crust and upper mantle.

Receiver responses remove much of the common earthquake source and distant propagation path from a multicomponent recording. They are used to identify major subsurface interfaces, investigate crustal thickness, and compare anisotropic or dipping structures. Finite bandwidth, interfering arrivals, and instrument noise limit the detail that can be inferred from any single event.

## Problem Description

The calibrated vertical, north and east ground velocities are denoted by Z, N and E. Rotate the horizontal observations using the known station-to-source back azimuth b:

```math
R=-N\cos b-E\sin b,\qquad T=N\sin b-E\cos b.
```

R is radial motion away from the source; T is the transverse component in this stated convention. In the plane-wave approximation, the vertical trace s acts as a measured source proxy. Each horizontal trace d is modeled by a receiver response h and noise e:

```math
d[n]=\sum_m s[m]h[n-m]+e[n].
```

Here n and m are integer sample indices, h is a discrete dimensionless convolution coefficient, and the sum is discrete without an extra sample-interval factor. The acquisition is one four-minute record at Walferdange, Luxembourg, during the 2011 Tohoku earthquake. No true crustal response is supplied. The vertical component is a source proxy rather than an exact incident-wave measurement; converted phases, scattered energy and instrument calibration uncertainty limit interpretation.

Save `output/reconstruction.npz` relative to the task workspace, containing `reconstruction`: real float32 or float64, finite, shape **(1, 2, 4801)**. Axes are batch, component (R then T), lag sample. Values are signed dimensionless receiver amplitudes. Sample k represents lag k/20 − 30 seconds. Use a 0.5 Hz Gaussian response bandwidth and normalize both components by the positive peak of the corresponding vertical self-response, so its peak is one; no independent component rescaling, phase alignment, translation, or cropping is applied by evaluation. The full declared arrays are evaluated separately for radial and transverse cosine similarity and dynamic-range-normalized RMS difference. This assesses agreement with the specified bandwidth and amplitude convention; it does not establish a unique true Earth model.

## Data Description

The NPZ contains instrument-response-corrected measured velocities, retaining the native sampling and float64 calibration precision.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| waveforms | (1, 3, 4801) | float64 | Ground velocity, m/s; components vertical upward, north, east; uniformly sampled time increasing along the last axis. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sampling rate | 20.0 | Hz | Original uniformly sampled record |
| Sample interval | 0.05 | s | Spacing between consecutive samples |
| Acquisition start | 2011-03-11T05:56:29.992872Z | UTC | Native sample nearest the fixed requested start |
| Duration | 240 | s | First-to-last sample interval; both endpoints retained |
| Station | GE.WLF | dimensionless identifier | Walferdange, Luxembourg |
| Station latitude | 49.6646 | degree north | Public station coordinates |
| Station longitude | 6.1526 | degree east | Public station coordinates |
| Station elevation | 295 | m above sea level | Site elevation |
| Sensor local depth | 80 | m | Borehole/installation depth |
| Back azimuth b | 33.06338963947741 | degree clockwise from north | Station-to-epicenter direction |
| Event latitude | 38.297 | degree north | Tohoku epicenter used for direction |
| Event longitude | 142.373 | degree east | Tohoku epicenter used for direction |
| Velocity calibration | Complete public StationXML response removed | m/s | Channel-specific frequency responses removed before packaging |
| Calibration frequency prefilter | 0.02, 0.03, 2, 4 | Hz | Four cosine-prefilter corners used during response removal |
| Calibration water level | 60 | dB | Instrument deconvolution stabilization |
| Calibration edge handling | Mean removal; 5% cosine taper on each side | dimensionless | Applied once during velocity calibration |
| Noise | Unknown, colored | m/s | No known independent Gaussian variance is asserted |
| Spatial boundary | Single station; plane-wave source approximation | dimensionless | No spatial periodicity assumption |
| Time boundary | Finite measured trace, zero extension | dimensionless | FFT implementation may use 4860 points; physical observations remain 4801 samples |

## Method Hints

Regularized spectral deconvolution avoids dividing directly by weak source frequencies. A water-level family clips source power below a fraction of its peak and applies a Gaussian low pass:

```math
\widehat H(f)=\frac{G(f)D(f)S(f)^*}{\max(|S(f)|^2,\alpha\max_\nu |S(\nu)|^2)},\qquad G(f)=\exp[-f^2/(2\sigma^2)].
```

S and D are the discrete Fourier transforms of the source and horizontal response, the star means complex conjugation, f and ν are frequency in Hz, α is a positive stabilization fraction, and σ sets the Gaussian bandwidth. A time delay places negative and positive response lags in one output array. A common normalization derived from the source self-response preserves relative component amplitudes. Detrending and tapering reduce finite-window spectral leakage. Time-domain spiking deconvolution and iterative sparse-spike deconvolution offer useful comparisons for the stability of conversion arrivals.

For the declared response convention, use α = 0.05 and σ = 0.5 Hz on a 4860-point FFT grid. The finite-record preparation uses linear detrending, a 5% Hann taper at each edge, and a causal fourth-order 0.05–1 Hz Butterworth bandpass. These choices define the finite-band operational response rather than a uniquely identifiable true crustal impulse response.

## References

- Eulenfeld, T. (2020). *rf: Receiver function calculation in seismology*. JOSS 5(48), 1808. https://doi.org/10.21105/joss.01808
- Ligorria, J. P., and Ammon, C. J. (1999). *Iterative deconvolution and receiver-function estimation*. BSSA 89(5), 1395–1400. https://doi.org/10.1785/BSSA0890051395
- GEOFON Data Centre (1993). *GEOFON Seismic Network*. https://doi.org/10.14470/TR560404 . Waveforms licensed CC BY 4.0; instrument calibration and stated acquisition selection applied.
