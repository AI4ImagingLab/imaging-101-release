# NADH component lifetimes and bound fraction

> Recover spatial free and enzyme-bound NAD(P)H lifetimes and bound intensity fractions from measured time-resolved cellular fluorescence histograms with overlapping decay components.
> Domain: biology | Keywords: fluorescence imaging, multicomponent inverse problems, harmonic analysis | Difficulty: Hard

## Background

Fluorescence lifetime microscopy records the delay distribution of emitted photons after pulsed excitation. The decay time depends on molecular environment and can reveal chemical binding states even when emission spectra overlap. Endogenous NAD(P)H fluorescence provides a label-free view of cellular heterogeneity relevant to metabolism, cancer biology and treatment monitoring.

Free and enzyme-bound populations can contribute within the same image pixel. Their decays overlap, and the instrument shifts and broadens measured timing distributions. A separate fluorescent standard provides timing-response calibration. Component lifetimes and their relative fluorescence contributions carry different information from a single apparent lifetime. Intensity fractions are not molecular concentration fractions or direct measurements of metabolic flux.

## Problem Description

The observation is a processed time-correlated fluorescence histogram at each pixel of human breast cancer cells. Model its normalized calibrated harmonic moment as

```math
z_h(p)=\frac{1-f(p)}{1-\mathrm{i}h\omega\tau_f(p)}+\frac{f(p)}{1-\mathrm{i}h\omega\tau_b(p)},\qquad h=1,2.
```

Here pixel index is p, imaginary unit is i, angular frequency is omega, harmonic index is h, fast and slow component lifetimes are tau-f and tau-b in ns, and f is the slow-component fluorescence intensity fraction. The two components represent a biexponential approximation to free and bound NAD(P)H. A component contributes its time-integrated intensity, not its initial exponential amplitude. For exponential amplitudes A-f and A-b, the bound intensity fraction equals A-b times tau-b divided by the sum of both amplitude-lifetime products.

For a histogram y with B bins, its uncalibrated moment is the sum of y at bin j times exp(2 pi i h j/B), divided by the histogram sum. The instrument correction multiplies this moment by the theoretical standard moment divided by the measured standard moment. Pool the complete independent standard image by intensity. The theoretical standard moment is 1/(1-i h omega tau-c), with known calibration lifetime tau-c. The acquisition table defines frequency, bin order and lifetime units.

The inverse problem estimates three unknowns per observed pixel from two complex moments. Noise, preprocessing and model mismatch make this nonlinear problem ill-conditioned; short-lifetime priors matter. No second species endpoint is provided as calibration. Pixels with zero observed histogram sum are outside the acquired cell support.

Save **output/reconstruction.npz** relative to the task workspace, with both required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 2, 512, 512) | float32 or float64 | ns; axes batch, component, row, column; component 0 is positive fast lifetime, component 1 is slow lifetime at least as large |
| bound_fraction | (1, 512, 512) | float32 or float64 | dimensionless slow-component integrated-intensity fraction in [0,1]; axes batch, row, column |

Both arrays must be real, finite, in the stated physical units and identically zero outside the observed support. Evaluation compares both arrays on every pixel whose observed histogram sum is positive, without alignment, cropping or fitted scale. It uses dynamic-range-normalized RMS error and maximum absolute error for each quantity, together with a two-harmonic complex forward RMS residual. Method-reference agreement measures reproducibility; it does not establish biological truth.

For the selected discrete graphical model, candidate fast lifetimes form the mathematical prior grid {0.20 + 0.01 k ns : k=0,...,60}. Minimize second-harmonic squared perpendicular distance over admissible grid-component pairs, and report the second-harmonic projection as the bound fraction. Equivalent minima are accepted. One of the two lifetimes must be within 1e-7 ns of the prior grid; the first-harmonic line residual and fraction-projection discrepancy must each be at most 1e-7. At every observed pixel the second-harmonic squared distance may exceed the grid minimum by at most 1e-12. When component phasors coincide within squared distance 1e-20, their separate fractions are unidentifiable and are not compared. The overall two-harmonic complex RMS residual must also be at most 0.037770825912687135. These tolerances define numerical geometric accuracy, without preferring an implementation-dependent tie.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 512, 512, 97) | uint8 | Author-provided processed cellular TCSPC histogram; arbitrary stored intensity units; axes batch, row, column, delay bin |
| calibration | (1, 512, 512, 97) | uint8 | Independent fluorescein-in-ethanol histogram in identical stored intensity units and axis order |

These are complete author-provided images, with no spatial cropping or additional filtering. The cell image was already masked and filtered before publication. Stored intensities are not unprocessed photon counts; a Poisson likelihood for these uint8 values is not justified.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Repetition frequency | 80 | MHz | omega = 2 pi times 0.08 per ns |
| Period | 12.5 | ns | Periodic moment convention |
| Delay bins | 97 | bins | Full supplied axis; uniform phase samples j/97 for j=0,...,96 |
| Standard lifetime | 3.05 | ns | Fluorescein in 100% ethanol, pH 6.5 |
| Image extent | 512 by 512 | pixels | Row down, column right; no spatial boundary padding |
| Field of view | 150 by 150 | micrometers | Data-paper acquisition context |
| Excitation wavelength | 750 | nm | Two-photon excitation |
| Emission band | 451 / 106 | nm | Bandpass center / width |
| Acquisition power | 25 | mW | Data-paper context |
| Frame averaging | 20 | frames | Data-paper context; already processed |
| Pixel dwell time | 8 | microseconds | Per frame |
| Frame rate | 0.20 | frames/s | Data-paper context |
| Histogram phase origin | first supplied bin | bins | Unknown instrument phase corrected by independent standard |
| Noise variance | unspecified | stored units squared | Processing prevents an exact photon-noise model |

The TIFF sources lack embedded acquisition metadata; scalar settings above come from the accompanying published protocol and official analysis example. Uniform phase sampling is the published implementation's numerical convention; an absolute timing-bin calibration beyond that convention is unavailable.

## Method Hints

Multi-harmonic phasor decomposition exploits the universal semicircle of a monoexponential decay. A trial fast lifetime fixes a point on that semicircle; the line through the first-harmonic observation determines the second component. Mapping both component points to the second harmonic allows the distance to its measured point to rank candidate pairs. Projecting onto the selected component line gives relative integrated intensities. A physiologically motivated fast-lifetime range stabilizes the blind search. 

For a semicircle real coordinate g at the first harmonic, its second-harmonic real coordinate is g/(4-3g); if that mapped real coordinate is G, its positive imaginary coordinate is sqrt(G-G squared). Finite noise means the two harmonics need not yield identical fractions. Diagnose forward residuals and prior sensitivity rather than interpreting all fitted lifetimes as independently validated molecular constants. Single-harmonic analysis with an assumed free lifetime supplies a useful comparison but does not uniquely identify all three unknowns per pixel.

## References

- Vallmitjana et al., *Blind Resolution of Lifetime Components in Individual Pixels of Fluorescence Lifetime Images Using the Phasor Approach*, J. Phys. Chem. B 124 (2020). [Theory](https://doi.org/10.1021/acs.jpcb.0c06946).
- Georgakoudi et al., *Consensus guidelines for cellular label-free optical metabolic imaging: ensuring accuracy and reproducibility in metabolic profiling*, J. Biomed. Opt. 30(S2), S23901 (2025). [Data and acquisition context](https://doi.org/10.1117/1.JBO.30.S2.S23901).
- Janet Sorrells, [Cell image](https://doi.org/10.6084/m9.figshare.28075163.v1) and [fluorescein reference](https://doi.org/10.6084/m9.figshare.28072340.v2), 2024, CC BY 4.0. Conversion preserves every value and transposes axes into batch-first NPZ format.
