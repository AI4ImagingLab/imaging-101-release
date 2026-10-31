# FRB Dispersion and Intrinsic Morphology

> Recover a burst's dispersion offset, intrinsic duration, arrival time, amplitude and curved radio spectrum from a real incoherently dedispersed dynamic spectrum whose finite channel widths smear a narrow pulse.
> Domain: astronomy | Keywords: radio transient spectroscopy, plasma propagation, pulse morphology | Difficulty: Hard

## Background
Radio bursts travel through ionized material, which delays lower frequencies more strongly. Their time-frequency structure can therefore probe plasma columns while also describing the intrinsic radio-emission morphology. A dynamic spectrum records intensity as a function of frequency and observing time.

Finite frequency channels retain dispersion smearing even after channel centers have been aligned. Intrinsic duration, spectral curvature and residual dispersion can trade off when a pulse spans only a few time samples. Joint modeling of frequency and exposure integration helps separate these effects under an explicit pulse-shape assumption.

## Problem Description
The observation is the complete author-provided CHIME/FRB Stokes I spectrum of FRB20220502B. Select the published conditional single-Gaussian branch with unresolved scatter broadening fixed to zero. This is a pulse-model assumption, not a measurement that the true scattering time vanishes. Retain all native measurements; evaluate every sample in all 10712 provided good frequency channels.

Let nu(k) be a channel center in MHz, t(n) a bin center in seconds relative to the supplied time origin, and nu(r) the spectral reference frequency. Known D0 is the dispersion measure already removed incoherently. Infer residual deltaD in pc/cm3; the absolute model dispersion is D=D0+deltaD. For frequency nu inside channel k, the arrival time is

```math
T_k(\nu)=t_0+K(D_0+\delta D)(\nu^{-2}-\nu_r^{-2})-KD_0(\nu_k^{-2}-\nu_r^{-2}).
```

Here K has units MHz²s cm³/pc and t0 is the intrinsic Gaussian center at nu(r), in relative seconds. The last term removes only the alignment already applied at the native channel center; it does not remove intra-channel smearing. With intrinsic standard deviation sigma>0, logarithmic amplitude alpha, spectral index gamma and spectral running beta, the underlying intensity is

```math
G_k(\nu,t)=10^\alpha\exp\!\left[\gamma\ln(\nu/\nu_r)+\beta\ln^2(\nu/\nu_r)\right]
\exp\!\left[-\frac12\left(\frac{t-T_k(\nu)}{\sigma}\right)^2\right].
```

Recorded intensity is approximated by a uniform 8-frequency by 4-time midpoint average of G over each rectangular channel/exposure bin. Both dimensions are averaged, not summed. These are the calibrated response discretization conventions; more accurate integration may be used if it matches this response to justified numerical accuracy. The source branch has no additive fitted background or stochastic per-channel amplitude. Observed signed intensities include noise; do not take their magnitude or subtract a newly fitted baseline.

Amplitude 10^alpha is the unaveraged intrinsic peak intensity at the reference frequency, in native instrument units U. No absolute Jansky calibration is supplied. The intrinsic Gaussian fluence at that frequency is10^alpha sigma sqrt(2pi), in U seconds. At other frequencies it is multiplied by the running-power-law spectral factor. Sigma is Gaussian standard deviation, not FWHM; FWHM=sigma sqrt(8ln2).

Save `output/reconstruction.npz` with key `reconstruction`, real float64 shape (1, 6), axes batch, parameter. Parameter order is alpha (dimensionless logarithm relative to 1 U), t0(second), sigma(second), deltaD(pc/cm3), gamma(dimensionless), beta(dimensionless). Require sigma>0 and D0+deltaD>0. All six quantities enter the complete forward prediction. No parameter permutation, arrival alignment, amplitude rescaling, frequency crop or hidden identity matching is applied.

Evaluation uses mean(((data-model)/RMS)^2) over all 162 time bins and all 10712 good channels. Let the visible good-channel set be S, its cardinality be 10712, y(c, t) be measured intensity, H(c, t) the full forward prediction, and r(c) the supplied noise RMS, all intensities in U. The dimensionless statistic is

```math
Q=\frac{1}{N}\sum_{c\in S}\sum_{t=0}^{161}\left(\frac{y(c,t)-H(c,t)}{r(c)}\right)^2,\qquad N=10712\times162.
```

 Per-channel RMS is the square root of mean native squared intensity over time indices 0..49, with no mean subtraction. No degrees-of-freedom subtraction occurs in this mean statistic. Physically different parameter vectors are accepted when their complete predicted data fit is equally good or better; fitted width/fluence tradeoffs need not have a unique solution.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements |(1,16384,162)|float64|Native signed Stokes I instrument intensity, U; batch, frequency, time; all RFI channels retained |
| frequencies |(1,16384)|float64|Native channel-center labels, MHz; ascending frequency |
| times |(1,162)|float64|Native bin-center labels, second relative to source time origin |
| good_channels |(1,16384)|bool|Dimensionless used-channel mask:provided RFI exclusion plus finite/nonzero-RMS checks; true means used |
| noise_rms |(1,16384)|float64|Off-pulse RMS intensity, U; computed from visible first 50 bins; unused channels may have zero RMS |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Applied incoherent dispersion D0 |623.6785965856119|pc/cm3|Already applied alignment from acquisition logistics, not the author's fitted DM |
| Dispersion constant K |4149.377593360996|MHz²s cm³/pc|Preserved source community convention |
| Dispersion exponent |-2|1|Cold-plasma arrival law |
| Spectral reference nu(r) |400.1953125|MHz|Fixed arbitrary spectral/arrival reference |
| Frequency spacing |0.0244140625|MHz|Native rectangular channel width |
| Native first leading-edge label |400.2197265625|MHz|Source file metadata; source reader adds half a channel |
| First channel center |400.23193359375|MHz|Metadata edge plus half width; public center array is authoritative |
| Time spacing/exposure width |0.0009830400813370943|s|Native time-bin width |
| Source time-origin label |1060314.72377856|s in source clock convention|Relative array starts at 0; do not fit this clock origin |
| Frequency sub-bin offsets |(-1/2+(j+1/2)/8) times channel width, j=0..7|MHz after multiplying width|Midpoint response quadrature |
| Time sub-bin offsets |(-1/2+(j+1/2)/4) times exposure width, j=0..3|s after multiplying width|Midpoint exposure quadrature |
| Intrinsic pulse/scattering |Single Gaussian; scattering fixed at zero|1|Conditional published model branch |
| Background |0 additional offset|U|No additional fitted/subtracted baseline in this task |
| Noise calibration interval |Native time indices 0..49|bin|Off-pulse RMS from visible observations |

Channel/time axes and observations retain native float64 precision. Frequency labels follow the archived source reader's edge-to-center convention, without rounding them to nominal telescope band limits. The supplied mask is observed/acquisition quality control; no fitted residual or hidden pulse parameter determines it. All bad-channel measurements remain in the visible array but have zero likelihood weight. No fitted burst-parameter cache is included.

## Method Hints
Use a joint nonlinear least-squares fit of cold-plasma timing and a curved spectral energy distribution. Integrating within each finite channel is essential at this dispersion. Test the no-residual-dispersion and no-spectral-curvature alternatives to reveal parameter coupling; compare predicted measurements rather than requiring one finite optimizer path.

## References
E.Fonseca et al., Modeling the Morphology of Fast Radio Bursts and Radio Pulsars with fitburst, Astrophysical Journal Supplement Series 271, 49 (2024), DOI 10.3847/1538-4365/ad27d6; arXiv 2311.05829v2.
