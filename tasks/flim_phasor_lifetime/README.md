# Calibrated Fluorescence Lifetime Imaging

> Reconstruct phase and modulation fluorescence lifetime maps from real photon arrival histograms despite instrument timing distortion and mixed fluorophore decays.
> Domain: biology | Keywords: fluorescence lifetime imaging, quantitative microscopy | Difficulty: Medium

## Background

After pulsed excitation, fluorescent molecules remain in excited states for a distribution of times before emitting photons. The fluorescence lifetime reflects the rates of radiative and nonradiative relaxation. It can change with local molecular interactions, oxygenation, chemical environment and energy transfer, providing contrast beyond fluorescence brightness.

Time-resolved microscopy records emission delay distributions at image pixels. It supports studies of cellular metabolism, molecular binding and tissue autofluorescence. Photon scarcity and instrument timing response make quantitative interpretation challenging, particularly when several emitting species contribute to one pixel.

## Problem Description

The observations are a native-resolution Convallaria microscopy image and a separate fluorescein calibration measurement. At each spatial pixel, the expected periodic histogram is proportional to the circular convolution of the instrument response with the fluorescence decay, sampled over one laser period. For delay index k, denote measured counts by h(k), the number of time bins by N, and normalized first-harmonic coordinates by g and s:

```math
g=\frac{\sum_{k=0}^{N-1}h(k)\cos(2\pi k/N)}{\sum_k h(k)},\qquad
s=\frac{\sum_{k=0}^{N-1}h(k)\sin(2\pi k/N)}{\sum_k h(k)}.
```

Let the calibration phasor be zc and the sample phasor z=g+is, where i is the imaginary unit. The ideal calibration phasor for known lifetime tc at angular frequency w is zt=1/(1-i w tc). The corrected sample phasor is Z=z zt/zc=G+iS. The two apparent lifetimes are defined as

```math
\tau_{\phi}=\frac{S}{\omega G},\qquad
\tau_{m}=\frac{\sqrt{(G^2+S^2)^{-1}-1}}{\omega},\qquad
\omega=2\pi f\,10^{-3}.
```

Here f is in MHz, w is in inverse ns, and all lifetimes are in ns. A multiexponential decay generally has different phase and modulation apparent lifetimes; neither is claimed to be a unique component lifetime. Recover both physical summary maps. Photon fluctuations and the unknown instrumental response prevent a direct interpretation of raw histogram shape as lifetime.

Save `output/reconstruction.npz` relative to your workspace with required real numeric arrays `reconstruction` (phase lifetime) and `modulation_lifetime`, each float64, shape `(1, 256, 256)`, axis order `(batch, y, x)`, in ns at their physical scale. Set both maps to zero outside `valid_mask`. No cropping, intensity rescaling, spatial alignment or phase alignment is performed. Evaluation uses uncentered cosine NCC and root mean squared error divided by the evaluated reference's dynamic range for every required map. It also checks RMS agreement between the calibrated observed phasor coordinates and those implied jointly by the two reconstructed apparent lifetimes on the valid support.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| measurements | (1, 256, 256, 256) | uint16 | Native spatial image photon counts; axes batch,y,x,delay; unit photons per delay bin accumulated over 40 frames |
| calibration | (1, 1, 1, 256) | uint64 | Independently measured fluorescein photon histogram pooled over its entire spatial image; unit photons per delay bin |
| valid_mask | (1, 256, 256) | uint8 | Fixed observation-validity support, 1 valid and 0 excluded; dimensionless |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Laser period | 24.99989318828099 | ns | Stored acquisition value; periodic temporal boundary |
| Fundamental frequency | 40.00017089948058 | MHz | Reciprocal period in matching units |
| Time bins | 256 | dimensionless | Uniform bins k=0,...,255 with t=k period/256 |
| Calibration lifetime | 4.1 | ns | Independently known fluorescein lifetime |
| Frames | 40 | dimensionless | Sample accumulation frames |
| Spatial grid | 256 by 256 | pixels | Original complete field of view, no resampling |
| Spatial coordinates | y then x, zero based | pixels | Original stored detector indexing; physical pixel pitch is unavailable and not needed for independent per-pixel inversion |
| Noise | Photon counting | photons | No background or Gaussian variance correction is supplied |
| Validity convention | count sum at least 200; G>0; S>0; G²+S²<1 | dimensionless | Fixed support derived solely from observations and independent calibration, before scoring |

Calibration photons are summed before normalization, equivalently giving intensity-weighted calibration phasor averaging. No dark count, time shift, median filter or implicit histogram offset is prescribed.

## Method Hints

Use the normalized first Fourier harmonic and an independently measured lifetime standard to divide out the instrument response. A complex rotation and scale correct the measured phasors. Closed-form phase and modulation inversion then provide complementary apparent-lifetime maps without assuming every pixel contains one molecular species.

## References

- Digman et al., *The Phasor Approach to Fluorescence Lifetime Imaging Analysis*, Biophysical Journal 94, L14–L16 (2008), [doi:10.1529/biophysj.107.120154](https://doi.org/10.1529/biophysj.107.120154).
- Weber, *Resolution of the fluorescence lifetimes in a heterogeneous system by phase and modulation measurements*, Journal of Physical Chemistry 85, 949–953 (1981), [doi:10.1021/j150608a006](https://doi.org/10.1021/j150608a006). The two apparent lifetimes summarize the harmonic response; they are not a unique molecular decomposition.
- PhasorPy contributors, *PhasorPy*, [doi:10.5281/zenodo.13862586](https://doi.org/10.5281/zenodo.13862586), the selected modern calibrated-phasor standard implementation.
- Rossetta, *Convallaria FLIM dataset in FLIM LABS JSON format* (2025), [doi:10.5281/zenodo.15007900](https://doi.org/10.5281/zenodo.15007900), CC BY 4.0. Observations converted losslessly to NPZ; independent calibration spatially pooled.
