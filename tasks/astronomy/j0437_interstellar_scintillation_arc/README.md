# Interstellar Scintillation Screen Geometry

> Recover a plasma-screen distance–velocity combination from eight measured pulsar intensity spectra whose noisy interference arcs are obscured by flagged channels and instrumental flux variations.
> Domain: astronomy | Keywords: wave scattering, radio astronomy | Difficulty: Hard

## Background

Radio waves from a compact source follow multiple paths through irregular interstellar plasma. Interference between these paths produces intensity fluctuations in time and observing frequency. The pattern carries information about scattering geometry and relative transverse motion even when the individual scattering structures are unresolved.

Geometric path delays depend quadratically on scattering angle, while Doppler shifts depend linearly on that angle. Their relation creates parabolic features in the two-dimensional intensity power spectrum. Measuring these features supports studies of the ionized interstellar medium, pulsar astrometry and transverse dynamics. Finite observing spans, radio interference and propagation through multiple structures limit the information available from a single observation.

## Problem Description

Infer the effective geometric combination κ from eight real Parkes observations of pulsar PSRJ0437−4715 on2011-12-20, and from their concatenation in time. For a dominant thin scattering screen, let D be the source distance and s the fractional screen coordinate measured from the source toward the observer. Define the effective geometric distance De=D s(1−s). Let Vparallel be the component of the effective transverse velocity along the relevant scattering direction. The identifiable combination is κ=De/Vparallel²; distance and velocity are not separately identifiable here.

Let f be the differential Doppler frequency inHz, β the nonnegative Fourier coordinate conjugate to wavelength incycles/m, λ the wavelength in m, τ the differential geometric delay in s and c the speed of light. The thin-screen interference relation is

```math
\beta=\frac{c\tau}{\lambda^2}=\frac{\kappa}{2}f^2.
```

The observed intensity contains sinusoidal interference terms in wavelength and time, whose cyclic Fourier coordinates are β and f. This uses cycles rather than angular frequencies; no additional2π factor is applied. Recovering κ is difficult because the intensity also contains modulation of the source/instrument gain, flagged samples, finite-window sidelobes, noise and contributions from other scattered paths. κ can vary between segments as the effective velocity changes. The concatenated value is an effective measurement over the complete observing span.

Save `output/reconstruction.npz` with the required key `reconstruction`, a finite positive `float64` array of shape `(1, 9)`. Its unit is s²m⁻¹. Columns0–7 correspond to the eight segments in increasing time order; column8 is the all-eight concatenated observation. No rescaling, sign, phase or translation alignment is allowed. Evaluation assesses every column by its measured arc-power objective: predicted curvature must attain power within0.5dB of the maximum of the corresponding observation-derived profile. A discrepancy from a particular fitted peak is diagnostic only. The admissible curvature-search interval is2<=kappa/(2e6)<=400 in the native wavelength-spectrum curvature units; endpoint samples outside the actual finite profile are inadmissible. This retains high-power equivalent peaks without enforcing one interpolation-specific peak coordinate. A separate screen distance or velocity is not an accepted substitute.

## Data Description

Every original integration is retained in the visible arrays, before rejection of short leading exposures or flagged band edges. Channel order is ascending frequency; intensity axes are batch, frequency, time. The measurement file's relative flux scale is retained; an absolute mJy calibration is not supplied. Zeros are flags, not zero astronomical intensity. The selected measurement model uses the supplied clock/frequency calibration directly.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `intensity_0` | `(1, 512, 122)` | `float64` | Segment0: Recorded intensity in relative ReferenceFluxDensity units; zero denotes flagged data |
| `frequency_0` | `(1, 512)` | `float64` | Segment0: Ascending radio-frequency channel centres, MHz |
| `time_0` | `(1, 122)` | `float64` | Segment0: Leading integration time relative to the header epoch, s |
| `mjd_0` | `(1, 1)` | `float64` | Segment0: Header start epoch as Modified Julian Date, day |
| `intensity_1` | `(1, 512, 121)` | `float64` | Segment1: Recorded intensity in relative ReferenceFluxDensity units; zero denotes flagged data |
| `frequency_1` | `(1, 512)` | `float64` | Segment1: Ascending radio-frequency channel centres, MHz |
| `time_1` | `(1, 121)` | `float64` | Segment1: Leading integration time relative to the header epoch, s |
| `mjd_1` | `(1, 1)` | `float64` | Segment1: Header start epoch as Modified Julian Date, day |
| `intensity_2` | `(1, 512, 122)` | `float64` | Segment2: Recorded intensity in relative ReferenceFluxDensity units; zero denotes flagged data |
| `frequency_2` | `(1, 512)` | `float64` | Segment2: Ascending radio-frequency channel centres, MHz |
| `time_2` | `(1, 122)` | `float64` | Segment2: Leading integration time relative to the header epoch, s |
| `mjd_2` | `(1, 1)` | `float64` | Segment2: Header start epoch as Modified Julian Date, day |
| `intensity_3` | `(1, 512, 122)` | `float64` | Segment3: Recorded intensity in relative ReferenceFluxDensity units; zero denotes flagged data |
| `frequency_3` | `(1, 512)` | `float64` | Segment3: Ascending radio-frequency channel centres, MHz |
| `time_3` | `(1, 122)` | `float64` | Segment3: Leading integration time relative to the header epoch, s |
| `mjd_3` | `(1, 1)` | `float64` | Segment3: Header start epoch as Modified Julian Date, day |
| `intensity_4` | `(1, 512, 121)` | `float64` | Segment4: Recorded intensity in relative ReferenceFluxDensity units; zero denotes flagged data |
| `frequency_4` | `(1, 512)` | `float64` | Segment4: Ascending radio-frequency channel centres, MHz |
| `time_4` | `(1, 121)` | `float64` | Segment4: Leading integration time relative to the header epoch, s |
| `mjd_4` | `(1, 1)` | `float64` | Segment4: Header start epoch as Modified Julian Date, day |
| `intensity_5` | `(1, 512, 121)` | `float64` | Segment5: Recorded intensity in relative ReferenceFluxDensity units; zero denotes flagged data |
| `frequency_5` | `(1, 512)` | `float64` | Segment5: Ascending radio-frequency channel centres, MHz |
| `time_5` | `(1, 121)` | `float64` | Segment5: Leading integration time relative to the header epoch, s |
| `mjd_5` | `(1, 1)` | `float64` | Segment5: Header start epoch as Modified Julian Date, day |
| `intensity_6` | `(1, 512, 121)` | `float64` | Segment6: Recorded intensity in relative ReferenceFluxDensity units; zero denotes flagged data |
| `frequency_6` | `(1, 512)` | `float64` | Segment6: Ascending radio-frequency channel centres, MHz |
| `time_6` | `(1, 121)` | `float64` | Segment6: Leading integration time relative to the header epoch, s |
| `mjd_6` | `(1, 1)` | `float64` | Segment6: Header start epoch as Modified Julian Date, day |
| `intensity_7` | `(1, 512, 121)` | `float64` | Segment7: Recorded intensity in relative ReferenceFluxDensity units; zero denotes flagged data |
| `frequency_7` | `(1, 512)` | `float64` | Segment7: Ascending radio-frequency channel centres, MHz |
| `time_7` | `(1, 121)` | `float64` | Segment7: Leading integration time relative to the header epoch, s |
| `mjd_7` | `(1, 1)` | `float64` | Segment7: Header start epoch as Modified Julian Date, day |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Source | PSRJ0437−4715 | identifier | Observed radio pulsar |
| Observatory | Parkes64m telescope | instrument | Radio observation site |
| Backend | CASPSR | instrument | Source file acquisition backend |
| Date | 2011-12-20 | UTC date | All eight observed segments |
| Segment order | 07:41:12,08:49:44,09:58:16,11:06:56,12:15:36,13:24:16,14:32:48,15:41:20 | UTC nominal start | Indices0–7; exact epochs are in MJD arrays |
| Native frequency channels | 512 | count | Full input channels per segment |
| Native integrations | 122,121,122,122,121,121,121,121 | count | Full input counts for segments0–7 |
| Nominal channel spacing | 0.78125 | MHz | Exact rounded channel centres are in frequency arrays |
| Nominal centre frequency | 1382 | MHz | Full-band centre before flagged-edge rejection |
| Typical integration interval | Approximately32 | s | Actual leading timestamps are supplied |
| MJD convention | JD−2400000.5 | day | Header epoch convention |
| Seconds per day | 86400 | s/day | Time conversion |
| Speed of light c | 299792458 | m/s | Wavelength/frequency conversion |
| Fourier convention | Cyclic, with phase2π(f t+β λ) | convention | f inHz; β incycles/m |
| Displayed Doppler unit | 1mHz=0.001Hz | conversion | Account for the squared conversion when reporting κ |
| Acquisition flag | Intensity0 | relative flux unit | Missing/flagged instrument sample |
| Scattering assumption | Dominant thin-screen interference | model assumption | Infer the geometric combination, not an absolute screen distance |

## Method Hints

Remove flagged edges and handle missing intensity samples without using any fitted screen parameter as calibration. A smooth gap model and a low-rank gain model can separate broad intensity variations from fine scintillation. Uniform-wavelength sampling helps remove chromatic curvature smearing. Compute the secondary intensity power spectrum, straighten trial parabolas by resampling each wavelength-conjugate row in Doppler, and estimate κ from the concentrated arc power. Retain all eight independent measurements and the concatenated observation; distinguish numerical agreement from physical uncertainty.

## References

- Reardon,D.J. etal.(2020). Precision orbital dynamics from interstellar scintillation arcs for PSRJ0437−4715. [DOI:10.3847/1538-4357/abbd40](https://doi.org/10.3847/1538-4357/abbd40).
