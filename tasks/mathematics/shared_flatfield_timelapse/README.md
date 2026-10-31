# Shared Flatfield from a Brightfield Time Series

> Recover a stationary sensor illumination field and a changing background level from moving biological images whose structure contaminates every frame.
> Domain: mathematics | Keywords: blind calibration, low-rank decomposition, robust estimation | Difficulty: Hard

## Background

An image records the product of the scene brightness and the spatial sensitivity of the imaging system. Uneven illumination and optical vignetting produce fixed spatial patterns. These patterns can obscure weak changes in a time series and create artificial differences between positions.

When the scene moves, the same sensor location sees different structures. The stationary illumination field can then be estimated jointly with the changing background. This form of blind calibration supports quantitative microscopy and longitudinal image comparison. It remains ambiguous when stationary scene structure resembles illumination or when the temporal background and spatial gain are allowed to exchange scale.

## Problem Description

The observations are 100 existing brightfield microscopy frames. Let $I_{tp}$ be frame $t$, resampled onto a $128\times128$ model grid at sensor location $p$, in digital numbers (DN). The conditional image model is

```math
I_{tp}=b_t S_p+R_{tp}.
```

Here $S_p>0$ is the dimensionless stationary flatfield, $b_t>0$ is the temporal background in DN, and $R_{tp}$ contains moving biological structure and measurement/model residuals. The additive darkfield is fixed to zero as an analysis assumption; it is not an independently measured calibration. Fix the scale ambiguity by $\frac1P\sum_pS_p=1$, where $P=128^2$. Neither absolute optical irradiance nor a uniquely identified biological foreground is requested.

Recover $S$ and all 100 values of $b$. The scientifically defined quality class balances the complete observation residual with spatial simplicity:

```math
\begin{aligned}
A(S,b)&=\frac{1}{100P}\sum_{t,p}|I_{tp}-b_tS_p|,\\
J(S,b)&=A(S,b)+\frac{5}{P}\sum_{k,l}
\left|[C(S\overline b)]_{kl}\right|,\qquad
\overline b=\frac1{100}\sum_t b_t .
\end{aligned}
```

The transform $C$ is the orthonormal two-dimensional type-II discrete cosine transform. For an axis of length $n=128$, its matrix entries are $C_{ki}=a_k\cos[\pi(i+1/2)k/n]$, where $a_0=n^{-1/2}$ and $a_k=(2/n)^{1/2}$ for $k>0$; the two-dimensional transform acts on both axes. Both $A$ and $J$ have units DN. Every model pixel of every frame contributes equally; there is no hidden foreground mask.

Save **output/reconstruction.npz** with these two real floating arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 128, 128) | float32 or float64 | Positive flatfield $S$, dimensionless, axes batch, sensor row, sensor column; mean one |
| baseline | (1, 100) | float32 or float64 | Positive $b$ in DN, axes batch, chronological frame index |

All values must be finite. Evaluation accepts any admissible pair with $J\le10.09925304298181$ DN, $A\le1.6162489661058033$ DN, and $|\operatorname{mean}(S)-1|\le10^{-5}$. It applies no rescaling or alignment. These are joint physical-quality criteria, not a requirement to reproduce a particular iterative trajectory or a hidden unique field.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| images | (1, 100, 1040, 1388) | uint8 | Original uncorrected brightfield images, DN; axes batch, chronological frame, sensor row, sensor column |

The native observations retain their original precision and dimensions. The smaller grid is the conditional model discretization, not a claim that the sensor acquired only 128 pixels per axis.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Modality | brightfield microscopy | — | Stem-cell time series |
| Frame order | t05100 through t05199 | frame identifier | All 100 successive supplied images |
| Sensor grid | 1040 rows × 1388 columns | pixel | Origin at upper-left; row increases downward and column rightward |
| Digitization | 0 through 255 | DN | Original unsigned 8-bit recordings |
| Physical pixel pitch / frame interval | not supplied | — | No physical length or time scale is inferred |
| Model grid | 128 × 128 | model pixel | Complete sensor field on the specified resampling grid |
| Model darkfield | 0 | DN | Fixed analysis assumption |
| Spatial boundary | truncated normalized triangle support | — | Resampling uses only available native pixels |
| DCT boundary | even extension at half-pixel edges | — | Orthonormal type-II transform |
| Noise | unspecified | DN | No Gaussian variance or photon gain is invented |

The resampling definition is part of the observation model. For either native axis of length $n$, set $s=(n-1)/127$. For output index $i=0,\ldots,127$, define $c_i=s(i+1/2)$, native indices $j=0,\ldots,n-1$, and

```math
w_{ij}=
\frac{\max(0,1-|j-c_i+1/2|/s)}
{\sum_{j'=0}^{n-1}\max(0,1-|j'-c_i+1/2|/s)} .
```

Apply these weights independently along rows and columns to native DN values. This is the antialiased triangle convention with aligned corners. The positive weights preserve constants and integrate the full sensor domain; values outside the sensor are absent. Float32 arithmetic is sufficient for resampling, with float64 accumulation appropriate for the quality functional.

## Method Hints

A robust low-rank background plus a sparse or heavy-tailed residual separates persistent sensor variation from moving objects. A cosine-domain sparsity penalty favors smooth illumination without requiring a fixed parametric lens profile. Alternating background estimation, proximal shrinkage, and residual reweighting are useful algorithm families. The fixed quality functional above is an evaluation criterion; an adaptively reweighted algorithm need not exactly minimize that fixed functional at each iteration.

## References

- Peng et al., “A BaSiC tool for background and shading correction of optical microscopy images,” Nature Communications 8, 14836 (2017), [doi:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836).
- This benchmark uses a modern approximate mathematical formulation with zero additive darkfield, conditioned on the explicit model and quality functional given above.
