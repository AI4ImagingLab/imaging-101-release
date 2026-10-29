# Mitchell GNSS Reflector Height

> Recover reflecting-surface heights from real satellite SNR interference arcs despite uneven elevation sampling, multipath noise and multiple spectral peaks.
> Domain: earth science | Keywords: reflectometry, inverse interference, spectral estimation | Difficulty: Medium

## Background

Navigation-satellite signals reach a ground antenna both directly and after reflection from nearby surfaces. Their phase difference changes as the satellite rises or sets. Interference modulates the recorded signal-to-noise ratio, carrying information about the vertical distance between antenna and reflecting surface.

GNSS reflectometry uses this otherwise unwanted multipath signal to monitor environmental surfaces. A single receiver can observe many satellite tracks and carrier bands. Uneven angular sampling, changing reflecting footprints and receiver noise create competing spectral peaks; inferred heights are conditional on a locally horizontal reflector model.

## Problem Description

For a measured arc let e be elevation angle, lambda the carrier wavelength, h reflector height, a and b unknown quadrature amplitudes, and n residual error. Define x=2sin(e)/lambda. The detrended interference model is

```math
y(x)=a\cos(2\pi hx)+b\sin(2\pi hx)+n(x).
```

Here y is relative-voltage SNR after author processing; x has inverse-metre units, h metres and a,b relative-voltage units. Different arcs have independent nuisance amplitudes and possibly different effective reflecting footprints. Recover h for every one of324 fixed-quality-controlled arcs from Mitchell station, Australia, days10--12 of2025, across GPS L1,L2C,L5. Observations are real author-processed arcs: refraction correction and direct-signal polynomial removal have already occurred. There is no independent physical height truth.

Save `output/reconstruction.npz` containing `reconstruction`, shape `(1,324)`, dtype float64, finite real heights in metres. Axes are batch then fixed arc order. The order matches counts, half_wavelength and arc_metadata. Values must exceed0.5m and not exceed8m. No registration, offset or scale alignment is used. Evaluation checks maximum absolute height discrepancy against an independently executed standard estimator; its numerical tolerance is not a physical accuracy claim.

## Data Description

`data/raw_data.npz` stores all324 retained arcs from the complete three-day author acquisition. Flattened samples are concatenated in arc order; counts delimit their ranges. Two additional candidate arcs failed the original fixed QC and are excluded before benchmark binding, independently of any cleaned result. These are processed observations, not raw receiver logs.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `elevation` | (1, 36123) | float64 | Refraction-corrected elevation angle, degree; flattened arc samples |
| `signal` | (1, 36123) | float64 | Author detrended measured SNR, relative voltage; flattened samples |
| `counts` | (1, 324) | int64 | Samples per arc, dimensionless integer; cumulative offsets split flattened arrays |
| `half_wavelength` | (1, 324) | float64 | Known carrier wavelength divided by two, m; one per arc |
| `arc_metadata` | (1, 324, 5) | float64 | Columns: day of year (day), satellite ID (dimensionless), frequency code (dimensionless), azimuth (degree), mean UTC time (hour) |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Station | MCHL, Mitchell, Australia | identifier | Measured GNSS receiver |
| Calendar | 2025 days10,11,12 | day of year | Complete supplied three-day record |
| Position | latitude-26.358904661, longitude148.144960505 | degree | Geodetic location |
| Antenna ellipsoid height | 534.591379 | m | Author station metadata, not reflector height |
| Carrier codes | 1=L1;20=L2C;5=L5 | dimensionless | GPS carrier identity |
| Carrier wavelengths | Supplied half_wavelength multiplied by2 | m | Independent known carrier scale |
| Observation convention | Corrected elevation, polynomial-detrended voltage SNR | relative voltage | Author processing retained without synthetic degradation |
| Source elevation windows | Direct signal fit5--30; selected interference5--25 | degree | Fixed original processing |
| Direct polynomial degree | 4 | dimensionless | Already removed by author preprocessing |
| Refraction | Standard Bennett using author independent meteorological parameters | convention | Already applied to visible angles |
| Azimuth coverage | 0--360 | degree | Original full configured azimuth range |
| QC | Elevation-span and duration checks plus minimum spectral peak-to-noise2.8 | convention | Original fixed acquisition QC; no cleaned-output selection |
| Temporal edges | Native finite days, with available adjacent-day overlap | convention | No invented day9/13 measurements; boundary tracks can be incomplete |
| Physical assumptions | Locally horizontal stationary reflector, one dominant interference height per arc | convention | No soil-water calibration or phase-center correction |
| Noise variance | Not calibrated | relative-voltage squared | Do not assume exact photon or Gaussian confidence levels |

## Method Hints

Use a harmonic least-squares spectrum on the uneven x coordinates. At a trial height, solve the two-amplitude linear fit; select the dominant explained-power peak across admissible heights. This separates nonlinear height from linear quadrature nuisance parameters. Compare different carrier groups and inspect measured-signal residuals because an isolated large peak does not establish a unique physical reflector.

## References

Larson (2024), Gnssrefl: an open source software package in python for GNSS interferometric reflectometry applications, GPS Solutions28,165. [Article](https://doi.org/10.1007/s10291-024-01694-8).
