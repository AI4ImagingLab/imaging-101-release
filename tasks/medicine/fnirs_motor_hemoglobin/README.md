# Motor-Cortex Hemoglobin from Near-Infrared Intensities

> Reconstruct relative oxygenated and deoxygenated hemoglobin time courses from a complete two-wavelength human motor recording despite overlapping absorption spectra and unknown absolute intensity baselines.
> Domain: medicine | Keywords: optical neuroimaging, spectral inversion, physiological imaging | Difficulty: Medium

## Background

Near-infrared light penetrates tissue through repeated scattering and is absorbed differently by oxygenated and deoxygenated hemoglobin. Measuring its attenuation at two wavelengths makes changes in these chromophores observable without ionizing radiation. The sensors are portable and can record while a participant moves or performs a behavioral task.

Functional near-infrared spectroscopy is used to study neurovascular responses, rehabilitation, and brain–computer interfaces. Continuous-wave instruments do not determine absolute optical pathlength or absolute concentration. Their concentration changes depend on an assumed partial pathlength factor and can include superficial scalp physiology as well as cerebral signals.

## Problem Description

The observations are all channels and samples of one real continuous motor recording. Recover the two effective concentration changes at each source–detector pair, retaining short-separation channels. The acquisition model is


```math
I_{pwt} = \bar I_{pw}\exp\left[-1000\,d_p f_w\sum_{j=1}^{2}E_{wj}c_{pjt}\right],\qquad
z_{pwt}=-\ln(I_{pwt}/\bar I_{pw}).
```


Here $p$ indexes source–detector pairs, $w$ wavelengths, $t$ samples, and $j$ the chromophores HbO (oxygenated) and HbR (deoxygenated). $I$ is measured intensity in instrument arbitrary units, and $\bar I$ is its arithmetic temporal mean separately for every pair and wavelength. The dimensionless optical density is $z$; $d$ is source–detector distance in meters; $f$ is the assumed dimensionless partial pathlength factor; $E$ is natural-log absorption in $\mathrm{m}^{-1}\mathrm{mM}^{-1}$; $c$ is relative concentration in mol/L (M). The factor 1000 converts M to mM. This mean-intensity convention does not force the temporal mean of concentration to zero. Spectral overlap causes cross-talk and pathlength uncertainty produces concentration-scale uncertainty. Two wavelengths cannot separate additional absorbers or depth compartments.

Save `output/reconstruction.npz` relative to the task workspace, containing exactly the required key `reconstruction`: a real finite `float64` array of shape `(1, 68, 2, 4860)` in M, ordered `(batch, pair, chromophore, time)`. Chromophore order is `(HbO, HbR)`; pair and time ordering match the inputs. Preserve signed concentration changes and the mean-intensity baseline. No temporal filtering, rescaling, spatial alignment, magnitude conversion, or channel omission is applied in evaluation. Each chromophore is assessed with cosine NCC (without centering) and RMS error divided by the reference dynamic range. Optical-density forward consistency is also checked as the relative Frobenius residual of the declared model. These criteria assess the specified effective inversion and do not establish absolute physiological accuracy.

## Data Description

`data/raw_data.npz` contains only observations and independent instrument/spectral calibration. All arrays are batch-first.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| intensity | (1, 68, 2, 4860) | float64 | Complete continuous-wave intensity recording, instrument arbitrary units; wavelength axis 760 then 850 nm. |
| distances | (1, 68) | float64 | Source–detector Euclidean separation in meters. |
| extinction | (1, 2, 2) | float64 | Natural-log absorption coefficients in m^-1 mM^-1; rows wavelength, columns HbO then HbR. |
| positions | (1, 68, 3) | float64 | Source–detector midpoint coordinates in meters, in the original head coordinate system; used to locate channels. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Dataset | FRESH Motor Dataset, subject 01, session left2s | — | Complete native recording, not a temporal or spatial crop. |
| Sampling frequency | 8.928571428571429 | Hz | Uniform native sampling rate. |
| Samples | 4860 | — | Time coordinates are sample index divided by sampling frequency, starting at zero. |
| Last sample time | 544.208 | s | Time coordinate of the final sample. |
| Wavelength order | 760, 850 | nm | Rows of the spectral calibration and wavelength axis. |
| Chromophore order | HbO, HbR | — | Columns of the spectral calibration and output species axis. |
| Partial pathlength factor | 6.0 at both wavelengths | dimensionless | Assumed effective pathlength multiplier, fixed for this inverse problem. |
| Intensity baseline | arithmetic temporal mean, per channel | arbitrary units | Reference intensity for natural-log optical density. |
| Noise model | unspecified measured instrument noise | — | No calibrated per-sample noise variance is supplied. |
| Spatial boundary | independent source–detector paths | — | No spatial boundary condition or depth-resolving model. |
| Coordinates | original head coordinates, x/y/z | m | Positions are channel midpoints; distances retain original optode separation. |

Data are by Rickson C. Mesquita and released under CC0. The assumed pathlength is a modeling convention, not a measured individual brain calibration.

## Method Hints

Use a modified Beer–Lambert spectral inversion. Optical density removes multiplicative channel intensity baselines. The two absorption spectra form a small linear system at every pair and time; a full-rank inverse or pseudoinverse separates HbO and HbR. Check spectral ordering, meter/millimeter conversion, logarithm base, and concentration units. Both recovered components are signed. Spatial or temporal priors would change the specified effective concentration estimator.

## References

- FRESH Motor Dataset, version 1.0.0, Rickson C. Mesquita. https://doi.org/10.18112/openneuro.ds005963.v1.0.0 (CC0).
- Gramfort et al. “MEG and EEG data analysis with MNE-Python.” Frontiers in Neuroscience 7, 267 (2013). https://doi.org/10.3389/fnins.2013.00267.
- Luke et al. “Analysis methods for measuring passive auditory fNIRS responses generated by a block-design paradigm.” Neurophotonics 8, 025008 (2021). https://doi.org/10.1117/1.NPh.8.2.025008.
- Prahl, “Tabulated Molar Extinction Coefficient for Hemoglobin in Water.” https://omlc.org/spectra/hemoglobin/summary.html.
