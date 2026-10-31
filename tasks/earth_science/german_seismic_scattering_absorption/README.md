# German Earthquake Scattering and Absorption Spectra

> Reconstruct frequency-dependent crustal scattering, intrinsic absorption, earthquake source energy and station amplification from measured three-component seismic records whose amplitude and decay contributions are coupled.
> Domain: earth science | Keywords: seismic imaging, radiative transfer, spectral inversion | Difficulty: Hard

## Background

Seismic energy spreads through the crust as direct waves and as a long coda created by repeated scattering from heterogeneity. Intrinsic absorption converts elastic energy into heat, while scattering redirects energy without necessarily dissipating it. Distinguishing these effects helps characterize the structure and physical state of the crust.

Recorded amplitudes also depend on earthquake radiation and local station geology. Joint interpretation of several stations and frequency bands can separate these contributions under an effective homogeneous transport model. Such estimates support regional attenuation characterization and comparisons of earthquake source spectra.

## Problem Description

The observations are calibrated, instrument-measured ground velocities from five German earthquakes and five regional stations, with 24 available event-station records. Reconstruct a separate effective transport spectrum for each earthquake together with its source spectrum and relative station responses. The medium is an infinite isotropic three-dimensional scattering medium with scalar propagation speed. The observation equation is

```math
E_{es}(t,f)=W_e(f)R_{es}(f)G(r_{es},t;g_e(f))\exp[-b_e(f)t]+\epsilon_{es}(t,f).
```

Here $E$ is three-component spectral energy density in J m$^{-3}$ Hz$^{-1}$, $e$ indexes events, $s$ indexes station slots, $f$ indexes frequency bands, $r$ is source-station distance in metres and $t$ is time after the catalogue origin in seconds. $W$ is source energy in J Hz$^{-1}$, $R$ is dimensionless site amplification, $g$ is the transport scattering coefficient in m$^{-1}$, $b$ is the intrinsic absorption rate in s$^{-1}$ and $\epsilon$ represents measurement noise and model mismatch. $G$ has units m$^{-3}$ and contains both a direct arrival and scattered coda. For propagation speed $c$, define $u=cgt$, $v=gr$ and $a=1-v^2/u^2$. Its scattered component for $ct>r$ is

```math
G_c=\frac{g^3a^{1/8}}{(4\pi u/3)^{3/2}}
\exp[u(a^{3/4}-1)]\sqrt{1+\frac{2.026}{ua^{3/4}}}.
```

The direct component is $\exp(-gr)\delta(t-r/c)/(4\pi cr^2)$, where $\delta$ is the time Dirac delta; both components vanish before the wavefront. This is the specified three-dimensional radiative-transfer approximation. The direct-arrival observation is a finite bulk-window mean; the coda is temporally smoothed. Source and site amplitudes have a multiplicative ambiguity, fixed by a geometric mean of one for the valid sites of each event-frequency pair. The mean-free-path and absorption contributions can remain correlated; estimates are conditional on the homogeneous-medium assumptions, not unique microscopic crustal properties.

Save `output/reconstruction.npz` relative to the task workspace with these required real floating-point arrays (float32 or float64):

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 5, 5, 2) | float32 or float64 | Batch, event, frequency, coefficient; last axis is positive scattering g in m^-1 then absorption b in s^-1 |
| source_energy | (1, 5, 5) | float32 or float64 | Batch, event, frequency; positive source energy W in J Hz^-1 |
| site_amplification | (1, 5, 5, 5) | float32 or float64 | Batch, event, frequency, event-specific station slot; positive dimensionless R for fit_valid slots, exactly zero otherwise |

Each output quantity is evaluated by the maximum absolute natural logarithm of its ratio to an independently reproduced standard-method result, without rescaling or alignment. A separate forward check evaluates the weighted logarithmic energy residual for every event-frequency pair. The site geometric-mean convention must hold; no absolute source calibration can be inferred independently of this convention.

## Data Description

The velocities retain the original calibrated float64 samples. Instrument sensitivity was removed using station inventory metadata before packaging. They are not synthetic waveforms or reconstructions. Zero-padded absent records must not be interpreted as observations.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| velocity | (1, 5, 5, 3, 4401) | float64 | Batch, event, event-specific station slot, component, time; calibrated velocity in m s^-1 |
| distance | (1, 5, 5) | float64 | Source-station catalogue distance in m; zero for absent records |
| start_time | (1, 5, 5) | float64 | First sample time relative to origin in s; zero for absent records |
| s_onset | (1, 5, 5) | float64 | Model S arrival in s after origin, stored at microsecond precision |
| frequency_bands | (1, 5, 2) | float64 | Lower and upper filter edges in Hz |
| valid | (1, 5, 5) | bool | Dimensionless mask of acquired event-station records |
| station_index | (1, 5, 5) | int64 | Dimensionless station identity index; -1 for absent records |
| fit_valid | (1, 5, 5, 5) | bool | Dimensionless event-frequency-slot quality mask determined only from observed coda duration |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Network | GR, German Regional Seismic Network | dimensionless | Regional broadband instrument records |
| Events in axis order | 20010623_0000004; 20020722_0000003; 20030222_0000013; 20030322_0000008; 20041205_0000033 | catalogue ID | Five actual earthquakes |
| Station identity order | GR.BUG; GR.TNS; GR.CLZ; GR.BFO; GR.FUR | dimensionless | station_index maps slots to this order |
| Component convention | Three orthogonal recorded ground-velocity components, summed as energy | dimensionless | Individual component signs do not change energy |
| Sampling rate | 20 | Hz | Already decimated upstream measurement records |
| Record interval | -10 to 210 inclusive | s | Relative to event origin, 4401 samples |
| Medium density | 2700 | kg m^-3 | Homogeneous effective density |
| Propagation speed | 3400 | m s^-1 | Fixed S-wave transport speed |
| Free-surface energy factor | 4 | dimensionless | Divide recorded energy by this factor |
| Frequency centres | 0.375, 0.75, 1.5, 3, 6 | Hz | Arithmetic means of supplied band edges |
| Frequency processing | Order-2 Butterworth bandpass, forward then reverse zero-state filtering | dimensionless | Zero phase, no endpoint extension, linear detrend first |
| Spectral energy convention | Sum rho times squared velocity plus squared Hilbert velocity, divided by twice effective bandwidth and free-surface factor | J m^-3 Hz^-1 | Effective bandwidth integrates the fourth power of single-pass filter gain |
| Background windows | [-10,-5] and [-5,0], endpoints included | s | Minimum window mean estimates stationary energy noise |
| Noise subtraction floor | Noise estimate / 100 | J m^-3 Hz^-1 | Positive energy floor after background subtraction |
| Bulk interval | S onset - 2 to S onset + 20, endpoints included | s | Mean energy; representative time is energy-weighted |
| Coda interval | S onset + 20 to min(S onset + 150, first crossing below 3 times noise) | s | Observed stopping rule; at least 0.05 s duration |
| Coda smoothing | 1-second Bartlett window, 20 samples | s | Nine left and ten right zero-padded samples for observed energy |
| Time sampling | Nearest sample, half sample rounded upward | dimensionless | Stored onset preserves original microsecond timestamp rounding |
| Medium boundary | Infinite homogeneous medium; free-surface correction applied to data | dimensionless | No reflecting boundary in transport Green function |

For effective bandwidth, integrate the single-pass squared gain twice over positive frequencies using a 512-point digital-frequency grid. The quadrature increment is sampling rate divided by 1024. The direct pulse is represented by a single time-bin density conserving its integrated energy. For each bulk mean use 1000 uniformly spaced time samples including endpoints. Coda Green functions use the same one-second smoothing window on an extended time grid. The first post-wavefront coda bin is omitted by the specified reference discretization, while the direct arrival occupies that bin if the grid crosses the wavefront. These choices define the discrete observation operator.

## Method Hints

Profile the nonlinear scattering coefficient while solving weighted linear least squares in logarithmic energy for station amplitudes and intrinsic absorption. The design contains one intercept per valid station and a common negative-time column. Give each coda sample unit weight and each bulk mean its original sample-count weight. Recover source amplitude from the mean fitted intercept, then relative site factors from deviations about that mean. A bounded scalar search is suitable for scattering; the absorption and energy parameters must remain positive. The weighted residual standard error uses the number of observations minus the number of stations minus one as its denominator. Actual regional records depart from the homogeneous model, so forward residuals should be interpreted as model consistency rather than a calibrated Gaussian noise statistic.

## References

- Eulenfeld and Wegler (2016), “Measurement of intrinsic and scattering attenuation of shear waves in two sedimentary basins and comparison to crystalline sites in Germany”, Geophysical Journal International 205, 744–757. https://doi.org/10.1093/gji/ggw035
- Sens-Schönfelder and Wegler (2006), “Radiative transfer theory for estimation of the seismic moment”, Geophysical Journal International 167, 1363–1372. https://doi.org/10.1111/j.1365-246X.2006.03139.x
- Paasschens (1997), “Solution of the time-dependent Boltzmann equation”, Physical Review E 56, 1135–1141. https://doi.org/10.1103/PhysRevE.56.1135
