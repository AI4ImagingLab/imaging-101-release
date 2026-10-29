# Calibrated E. coli Fluorescence Abundances

> Recover thirteen relative fluorophore-brightness images from a measured confocal spectral image whose broad, overlapping emission signatures and low signals obscure individual labels.
> Domain: biology | Keywords: fluorescence imaging, spectral unmixing, spatial regularization | Difficulty: Hard

## Background

Fluorescence microscopy detects light emitted after molecules absorb excitation light. Labels attached to biological targets make selected structures visible, but ordinary fluorophores emit across broad wavelength bands. A spectral detector records multiple bands at each location; separating their contributions allows more labels to be observed than a small set of conventional color filters permits.

Multiplex fluorescence is useful for locating cells, molecules and microbial communities. Neighboring pixels often belong to the same biological object, while an individual pixel contains only a small subset of the labels present across an experiment. Weak signals, detector quantization and substantial overlap between emission spectra make quantitative separation difficult even when independently stained controls are available.

## Problem Description

The input is a complete author-distributed 30×30-pixel region of a measured E. coli fluorescence image, with 24 detector channels and thirteen independently calibrated label spectra. It is an 8-bit processed microscope export, not calibrated absolute photon counts. Let the measured channel vector at pixel p be y(p), the fixed calibration matrix be M, and the nonnegative label-brightness vector be a(p). The acquisition model is

```math
\mathbb{E}[y(p)] = M a(p).
```

Here M has 24 rows and 13 columns, y has 24 entries in digital intensity units, and a has 13 coefficients in the corresponding digital brightness units. The calibration columns are normalized to unit maximum. A Poisson-like count fidelity can model signal-dependent fluctuations, with the limitation that quantization and prior export processing prevent a literal photon-count likelihood. Broad spectral overlap amplifies small perturbations. No per-pixel sum-to-one condition applies to brightness coefficients.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing `reconstruction`: real float32 or float64, shape **(1,30,30,13)**, axis order **batch, y, x, fluorophore**. Return nonnegative relative brightness, divided by the single largest coefficient over all pixels and labels so the global maximum is one. Units are dimensionless relative brightness; these are not molecular concentrations. The label order is AF488, AF514, TET, AF532, AF546, AF555, RRX, AF568, AF594, AF647, AF660, AF680, AF700. Set the one-pixel exterior border to zero because full neighborhoods are unavailable there. All arrays must be finite.

Evaluation version 2 uses the explicit regularized spectral objective defined below, rather than requiring agreement with a finite reweighted implementation. No spatial or label permutation alignment is used. An additional observation-consistency metric compares the interior reprojected spectra with measured spectra. For that metric only, the lost global brightness scale is restored by the nonnegative least-squares scalar

```math
\alpha = \max\left(0,\frac{\langle M\hat a,y\rangle}{\langle M\hat a,M\hat a\rangle}\right),
\qquad e=\frac{\|\alpha M\hat a-y\|_2}{\|y\|_2}.
```

The brackets are sums over all interior pixels and channels; hat a denotes the submitted relative abundance. A zero prediction has observation error one. This single scalar accounts for the explicitly discarded detector scale; it does not compensate for spatial or spectral errors.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `measurements` | (1,30,30,24) | uint8 | Measured author-distributed fluorescence ROI, y/x/channel order, digital intensity units from 0 to 255 |
| `spectra` | (1,24,13) | float64 | Independently stained control spectral responses, channel/fluorophore order, dimensionless with each column's maximum one |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Fixed, fluorescently labeled E. coli | — | Biological reference experiment |
| Instrument family | Zeiss confocal spectral microscope | — | Spectrally resolved fluorescence acquisition |
| Objective | Plan-Apochromat 63×/1.40 Oil DIC M27 | — | Objective identifier retained in export |
| Lateral pitch | 9.9521491860627815e-8 | m | Equal x and y sampling, native metadata precision |
| Excitation wavelengths | 488, 561, 633 | nm | Excitation lines described for the acquisition |
| Detector channels | 24, in stored order | — | Selected emission channels in the export |
| Emission interval | Approximately 491.874502–725.147638 | nm | Spectral coverage; calibration matrix defines the exact forward response |
| Channel bandwidth | Approximately 9.719714 | nm | Exported spectral-bin width |
| Bit depth | 8 | bit | Stored processed digital intensity; camera gain is not independently supplied |
| Intensity convention | Divide uint8 by 255 for unit-range computation | — | Numerical scaling, not a photon calibration |
| Coordinates | x increases by column, y increases by row | pixel | Top-left array origin; no image rotation |
| Boundary | Unreconstructed exterior one-pixel border equals zero | — | No periodic or reflected neighborhoods |
| Noise | Signal-dependent fluctuations; no independent variance calibration | — | Poisson-like modeling is an approximation for this processed export |

The region is supplied as distributed by the authors. It does not represent the entire microscope field of view.

## Method Hints

The v2 conditional target is a convex, nonnegative low-rank/group-sparse spectral fit. Flatten the 28 by 28 interior into columns, giving Y with 24 rows and 784 columns, divide digital intensities by 255 and then by their global root-mean-square value to obtain B. Let U be a nonnegative 13 by 784 brightness matrix and M the supplied spectral calibration. Define

```math
J(U)=\sum_{c,p}\left[(MU)_{cp}-B_{cp}
+B_{cp}\log\frac{B_{cp}}{(MU)_{cp}}\right]
+0.1\|U\|_*+0.1\sum_{j=1}^{13}\|U_{j,:}\|_2.
```

The nuclear norm is the sum of singular values. Terms with B=0 use zero for B log(B/prediction); positive B with zero prediction has infinite loss. This generalized KL fidelity is a signal-dependent digital-intensity modeling choice, not a literal calibrated photon-count likelihood. Both penalties act on the complete interior field, with no window normalization or inverse-singular-value reweighting. This is an explicit v2 change from the historical local reweighted SLPRU estimator, without a claim of mathematical equivalence.

Normalize the reported field to global maximum one. For evaluation, let a be this normalized 13 by 784 field, P=0.1(norm(a,nuclear)+sum of its row Euclidean norms), and alpha=sum(B)/(sum(Ma)+P). This scalar exactly minimizes J(alpha a), accounting for the discarded common brightness scale. Require J(alpha a)/(24*784)<=0.07310834324578482, an attainable standard-output quality bound. Also require the separate observation relative error defined above to be at most 0.3680738697471408. Both retain all channels and all interior pixels. More accurate objective minimizers are accepted without matching historical alternating updates.

Use calibrated nonnegative spectral regression with a count-aware data term. Small spatial neighborhoods can share a sparse set of fluorophores, and neighboring abundance vectors can have low rank. Splitting the forward prediction, low-rank constraint, group sparsity and nonnegativity gives tractable alternating updates. Compare count-aware fitting with nonnegative least squares to understand the tradeoff between spectral residual and spatial sparsity. Do not interpret a smaller residual alone as proof of better biological identification.

## References

- Wang et al. (2023). *Unmixing biological fluorescence image data with sparse and low-rank Poisson regression*. Bioinformatics 39(4), btad159. https://doi.org/10.1093/bioinformatics/btad159
