# Acceptor-photobleaching FRET efficiency in a cell membrane

> Reconstruct a spatial energy-transfer efficiency field from matched confocal images before and after acceptor bleaching, correcting donor bleaching, spectral cross-talk and fluorescent photoproducts.
> Domain: biology | Keywords: fluorescence microscopy, molecular interactions, quantitative imaging | Difficulty: Medium

## Background

An excited fluorescent donor can transfer energy without emitting a photon when a suitable acceptor lies within a few nanometres. This quenches donor emission and provides a sensitive probe of molecular proximity. Imaging the transfer efficiency can reveal how interactions vary across a cell membrane.

Destroying the acceptor by illumination removes much of this quenching. Comparing donor images before and after that intervention can recover a transfer-efficiency map. The inference is complicated by residual acceptor molecules, changes in donor brightness, spectral leakage and fluorescent products of the bleaching reaction. Separate singly labelled controls help distinguish these effects from energy transfer.

## Problem Description

The sample is a fixed SK-BR-3 cell preparation with two labelled epitopes of a membrane receptor. Three matched preparations are supplied: donor-only, acceptor-only, and doubly labelled. Each has donor and acceptor channels before and after acceptor bleaching. The scalar efficiency is a spatial ensemble measurement; it is not a unique molecular distance or a direct count of interacting receptors.

At a pixel, let the background-corrected donor measurements before and after bleaching be $D_1$ and $D_2$, and the acceptor measurements be $A_1$ and $A_2$. Let $D_0$ denote unquenched donor brightness and $E$ the transfer efficiency. The conditional acquisition model is

```math
D_1=D_0(1-E)+\delta A_1,\qquad
D_2=\frac{D_0(1-\alpha E)}{\gamma}
 +\bigl[\alpha\delta+(1-\alpha)\epsilon\bigr]A_1,
\qquad A_2=\alpha A_1.
```

Here $\gamma$ is the donor-only before/after brightness ratio, $\delta$ is acceptor leakage into the donor channel before bleaching, $\epsilon$ is the corresponding photoproduct ratio after bleaching, and $\alpha$ is the residual acceptor fraction. All four factors are dimensionless. The last relation defines an average residual fraction: individual noisy pixels need not satisfy it exactly. The model assumes one effective acceptor environment per donor and indiscriminate acceptor destruction. It does not identify arbitrary molecular stoichiometry or acceptor distributions.

Recover the standard fully corrected efficiency at the specified spatial smoothing scale, together with its four estimated correction factors. The fixed public support is determined from observed pre-bleaching signal, before efficiency estimation. Preserve negative efficiency estimates produced by measurement noise; do not clip them to a physical interval or discard pixels after fitting.

Save `output/reconstruction.npz` relative to the task workspace with exactly these real float32 or float64 arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | `(1, 512, 512)` | float32 or float64 | Dimensionless efficiency in batch, row-y, column-x order; exactly zero outside `observation_mask`, signed values retained inside. |
| correction_factors | `(1, 4)` | float32 or float64 | Dimensionless factors in the order gamma, delta, epsilon, alpha; retain the three-decimal factor convention. |

Every output value must be finite. Evaluation compares the complete supported field using cosine similarity, dynamic-range-normalized RMS error and maximum absolute error; the four factors also have an absolute-error check. There is no alignment, rescaling or clipping at evaluation time. This is a conditional standard-estimator agreement task, with numerical tolerances; agreement does not establish true molecular efficiency.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| observations | `(1, 3, 2, 2, 512, 512)` | uint16 | Measured intensity in ADU; axes batch, preparation, time, channel, row-y, column-x. Preparation order: donor-only, acceptor-only, doubly labelled. Time order: before, after bleaching. Channel order: donor, acceptor. Original 12-bit measurements are stored losslessly in uint16. |
| observation_mask | `(1, 512, 512)` | bool | Dimensionless fixed support for the doubly labelled field; true where both pre-bleaching channels exceed 100 ADU after the specified background and smoothing convention. This contains no efficiency information unavailable in the observations. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Microscope | Zeiss LSM510 confocal | — | Author-supplied real cell and matched-control observations. |
| Objective | 63× oil, NA 1.4 | — | Collection optics. |
| Donor label | Alexa Fluor555 | — | Donor-labelled receptor epitope. |
| Acceptor label | Cy5 | — | Acceptor-labelled second receptor epitope. |
| Donor excitation / detection | 543 / 560–615 | nm | Native donor track. |
| Acceptor excitation / detection | 633 / greater than650 | nm | Native acceptor track. |
| Detector gains | 732 donor, 652 acceptor | instrument setting | Matched across all three preparations; no invented cross-instrument gain. |
| Pixel spacing x, y | 0.0558035713935203 each | micrometre | Original LSM lateral sampling; x increases with columns, y with rows. |
| Time ordering | before, after acceptor bleaching | — | No quantitative kinetic time model is assumed. |
| Background measurement region | rows0–63, columns448–511, inclusive | pixel | Same observed background rectangle in each of the12 frames. |
| Registration geometry | integer translations of post-bleaching channels relative to pre-bleaching donor | pixel | The observed doubly labelled pair has zero cross-correlation shift. |
| Physical boundaries | independent finite field of view | — | No inferred signal beyond the field; smoothing extends nearest edge values. |
| Observation noise | not independently characterized | ADU | No known Gaussian or Poisson variance is supplied. |

## Method Hints

A corrected donor-quenching inverse separates unquenched donor brightness from energy transfer. Eliminating the brightness gives

```math
E=1-\frac{(1-\alpha)(D_1-\delta A_1)}
{\gamma\bigl[D_2-(\alpha\delta+(1-\alpha)\epsilon)A_1\bigr]
 -\alpha(D_1-\delta A_1)}.
```

Use ratio-of-mean intensities for the three independent singly labelled control factors. Donor-only before and after means each use their own donor signal above100 ADU; the acceptor-only before-channel support is used for both leakage and photoproduct ratios. Estimate the residual acceptor fraction as the mean of pixelwise after/before acceptor ratios on the fixed target support. Round the resulting factors to three decimal places, preserving the standard estimator's convention. A donor factor slightly below one is allowed and must not be silently replaced by one.

The requested conditional field uses the measured rectangle mean as background, with negative background-subtracted intensities set to zero, and a separable one-pixel Gaussian smoothing convention. Its symmetric one-sided weights at offsets0 through4 are `(0.3989434540271759, 0.2419714480638504, 0.053991127759218216, 0.004431861452758312, 0.0001338306174147874)`. These specify the finite kernel rather than an unspecified smoothing library default. The closed inverse should retain signed results and apply the fixed support only as the declared output convention.

## References

- Roszik J., Szöllősi J., Vereb G. *AccPbFRET: An ImageJ plugin for semi-automatic, fully corrected analysis of acceptor photobleaching FRET images.* BMC Bioinformatics9:346(2008). [Published method and real example observations](https://doi.org/10.1186/1471-2105-9-346).
- van Munster E.B. et al. *Fluorescence resonance energy transfer measurement by gradual acceptor photobleaching.* Journal of Microscopy218:253–262(2005). [Residual-acceptor model](https://doi.org/10.1111/j.1365-2818.2005.01483.x).
