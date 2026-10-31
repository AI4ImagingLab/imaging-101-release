# Four-Dimensional Transverse Beam Phase Space

> Recover a proton beam's four-dimensional transverse probability density from 24 real wire-scanner profiles transported through known accelerator optics.
> Domain: physics | Keywords: beam tomography, phase-space imaging, maximum entropy | Difficulty: Hard

## Background

An accelerator transports particles with a distribution of transverse positions and directions. That distribution determines beam size, losses and the illumination delivered to a target. A one-dimensional wire scan measures only the beam's projected transverse intensity, leaving correlations between position and direction hidden.

Linear magnetic optics rotate and shear phase space before a profile is measured. Combining horizontal, vertical and diagonal projections at multiple stations constrains the full four-dimensional distribution. Finite projection data still permit many densities, so an entropy condition supplies a conservative preference for a smooth prior while retaining measured non-Gaussian structure.

## Problem Description

The measurements are two experimental optical configurations at four wire-scanner stations, with three directions each: 24 profiles in total. Reconstruct the centered physical phase-space distribution in coordinates $x=(X,X',Y,Y')$, with positions in mm and paraxial directions in mrad. Centering is defined by the supplied per-profile acquisition statistics, rather than an unknown absolute beam orbit.

For transport group $k=0,\ldots,7$, the known matrix $T_k$ maps reconstruction-plane coordinates to a wire-scanner plane. Define direction rows

```math
d_x=(1,0,0,0),\quad d_y=(0,0,1,0),\quad
d_u=(1,0,-1,0)/\sqrt{2}.
```

The diagonal coordinate is $(X-Y)/\sqrt{2}$, with $X$ pointing right and $Y$ up when looking in the beam's direction of motion. The native vertical-wire positions are reflected as specified below.

First define the Gaussian-prior covariance $\Sigma$ from the supplied **raw RMS statistics**. These are processed observations from the same acquisition, not independent calibration or known density truth. Let $s_x,s_y,s_u$ denote the reported RMS widths for one transport group, and let $r_k,t_k$ be the first and third rows of its transport matrix. Define $\Sigma$ as the symmetric least-squares solution

```math
\mathop{\mathrm{argmin}}_{\Sigma=\Sigma^{\mathsf T}}
\sum_{k=0}^{7}\left[
(r_k\Sigma r_k^{\mathsf T}-s_x^2)^2+
(t_k\Sigma t_k^{\mathsf T}-s_y^2)^2+
\left(r_k\Sigma t_k^{\mathsf T}
-\frac{s_x^2+s_y^2}{2}+s_u^2\right)^2\right].
```

The supplied matrices have uncoupled horizontal/vertical transport; the distribution itself may have all cross-plane correlations. The design has full rank and yields positive-definite $\Sigma$. Its entries carry the products of the corresponding mm/mrad coordinate units.

Represent the unknown using a real invertible matrix $N$ and 35 equal cells per axis in $z=Nx\in[-4,4]^4$. Require $N\Sigma N^{\mathsf T}=I_4$. Thus each row of $N$ has coefficients in inverse mm for position columns and inverse mrad for angular columns. Different orthogonal whitening representations are allowed. A general rotation changes the finite cube and its cell partition; these are an allowed representation class, rather than an assertion that arbitrary rotated finite grids are identical.

Let $h=8/35$, let $c_v$ be the four-dimensional center of voxel $v$, and let $m_v$ be its probability. Define a piecewise-constant density:

```math
m_v\ge 0,\qquad\sum_v m_v=1,\qquad
f_z(z)=m_v/h^4\ (z\in v),\qquad
f_x(x)=|\det N|\,f_z(Nx).
```

The physical density $f_x$ has units $\mathrm{mm}^{-2}\mathrm{mrad}^{-2}$. Its recovered covariance is not forced to equal the prior covariance.

The analysis-profile centers are $s_j=-85+j\Delta$ mm, $j=0,\ldots,99$, with $\Delta=170/99$ mm. Analysis bin edges are $e_j=s_j-\Delta/2$ for $j=0,\ldots,99$ and $e_{100}=s_{99}+\Delta$. This deliberately preserves the asymmetric terminal edge of the selected histogram convention: widths $w_j=e_{j+1}-e_j$ equal $\Delta$ except the final width $1.5\Delta$. For direction $d$, define the physical bin probability and analysis density

```math
\pi_{kdj}=\int_{[-4,4]^4}f_z(z)\,
\mathbf{1}\!\left[e_j\le dT_kN^{-1}z<e_{j+1}\right]\,dz,\qquad
p_{kdj}=\frac{\pi_{kdj}/w_j}{\Delta\sum_{\ell=0}^{99}\pi_{kd\ell}/w_\ell}.
```

The last edge is included in the final bin. Integrating uniform four-dimensional cells produces a one-dimensional convolution of four uniform intervals, rather than point masses at cell centers. The normalization conditions profiles on the observed finite analysis window and matches the stated equal-center quadrature. All 24 acquired profiles are used.

Let $g_{kdj}$ be the normalized measured analysis profiles defined below. Each profile must meet its own relative residual

```math
\epsilon_{kd}=\frac{\|p_{kd}-g_{kd}\|_2}{\|g_{kd}\|_2}.
```

Because projections do not determine a unique four-dimensional density, also constrain the discrete relative entropy to the Gaussian midpoint prior:

```math
q_v=\frac{\exp(-\|c_v\|_2^2/2)}
{\sum_w\exp(-\|c_w\|_2^2/2)},\qquad
D(m\|q)=\sum_{v:m_v>0}m_v\log(m_v/q_v)\le 0.3381082948469032.
```

Here $q_v$ is explicitly a normalized midpoint prior on the same finite cells, not an asserted exact integral of a continuous Gaussian over each cell. The target is the class satisfying the physical, entropy and individual observation bounds; no particular random sampling trajectory is required.

| Transport group | Maximum $\epsilon_x$ | Maximum $\epsilon_y$ | Maximum $\epsilon_u$ |
| --- | --- | --- | --- |
| 0 | 0.026591262080207435 | 0.02844506149649671 | 0.018434185187322615 |
| 1 | 0.04025462873101504 | 0.027660589424938535 | 0.013654587954066176 |
| 2 | 0.0336181817804809 | 0.024701584356907547 | 0.026476045698105286 |
| 3 | 0.030282517003045205 | 0.02843132972019683 | 0.02053178642008002 |
| 4 | 0.037817523082168805 | 0.03739204205315514 | 0.014327218326556366 |
| 5 | 0.04814514228495083 | 0.018381331368848182 | 0.02633789514374982 |
| 6 | 0.017210314777749293 | 0.060767767261188405 | 0.03430500540870789 |
| 7 | 0.028060177656767955 | 0.020072433153777422 | 0.026829248297179618 |

Save output/reconstruction.npz with exactly these quantities:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 35, 35, 35, 35) | float32 or float64 | Real dimensionless voxel probabilities $m_v$; axes batch, increasing $z_0,z_1,z_2,z_3$; cell edges equally spaced from −4 to 4 |
| normalization | (1, 4, 4) | float32 or float64 | Real matrix $N$ mapping the mixed-unit column vector $(X,X',Y,Y')$ to dimensionless $z$; row then column order |

Every element must be finite. Evaluation checks every profile and the entropy condition, probability-sum error at most $10^{-6}$, maximum negative probability at most $10^{-12}$, and maximum absolute element of $N\Sigma N^{\mathsf T}-I_4$ at most $10^{-5}$. Tiny permitted probability roundoff is clipped and normalized for forward integration, while the submitted mass constraints are scored separately. There is no fitted translation, scaling, basis alignment or hidden density identity. Both required arrays jointly determine the physical distribution and are evaluated.

## Data Description

The file data/raw_data.npz preserves native float64 scanner positions and signal amplitudes. Profiles have different sample counts, so a visible contiguous validity mask distinguishes acquired samples from zero-filled format padding. Padding is not a zero measurement. Group order is first acquisition/stations 0–3, then second acquisition/stations 0–3; the transport matrices and statistics use that same order. Direction order is x, y, u.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| raw_coordinates | (1, 8, 3, 90) | float64 | Original wire coordinates in mm; axes batch, transport group, direction, native sample |
| raw_signal | (1, 8, 3, 90) | float64 | Original scanner signal in arbitrary acquisition units; same axes; padded entries zero |
| sample_mask | (1, 8, 3, 90) | bool | Dimensionless native acquisition validity; trailing false entries are format padding |
| profile_mean | (1, 8, 3) | float64 | Acquisition-reported raw RMS-statistics mean position in mm, used for centering |
| profile_sigma | (1, 8, 3) | float64 | Acquisition-reported raw RMS width in mm; squared values define the covariance observation |
| transfer_matrices | (1, 8, 4, 4) | float64 | Independent optics transport in mixed mm/mrad coordinates; each element has output-coordinate/input-coordinate units |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Measurement | SNS proton beam wire scans | — | Existing experimental acquisition |
| Acquisition start times | 2024-08-19 16:50:33 and 17:09:09 EDT | local date/time | Original acquisition headers |
| Configurations / stations / directions | 2 / 4 / 3 | dimensionless | Eight transport groups,24 profiles |
| Physical coordinate order | $X,X',Y,Y'$ | mm,mrad,mm,mrad | Centered reconstruction-plane coordinates |
| Diagonal direction | $(X-Y)/\sqrt 2$ | mm | Clockwise 45° direction |
| Analysis profile centers | −85 to 85,100 centers | mm | Spacing $\Delta=170/99$ |
| Terminal histogram edge | $85+\Delta$ | mm | Final bin width $1.5\Delta$ |
| Profile smoothing | Reflecting Gaussian, standard deviation 1 | analysis bin | Same observation-processing protocol for all profiles |
| Native-signal threshold | 0.01 times original profile maximum | signal units | Before smoothing; no output-derived mask |
| Adjacent-jump exclusion | Absolute jump greater than 0.25 times thresholded maximum | dimensionless fraction | Acquisition-processing outlier rule |
| Phase-space grid | 35 cells on each of 4 axes, cube[−4,4] | dimensionless normalized coordinates | Full four-dimensional representation |
| Prior | Isotropic Gaussian midpoint weights on the finite cells | dimensionless probability | Conditional regularization assumption |

To define the measured analysis profiles, use only true entries of sample_mask. Subtract the reported mean from native positions. For y, reverse samples and negate positions. If a signal contains negative values, shift its minimum to zero; threshold it using 0.01 times its original unshifted maximum. Remove each sample after the first whose absolute adjacent thresholded-signal jump exceeds 0.25 times the thresholded maximum. Interpolate linearly to the 100 centers with zero outside the native range, then apply a Gaussian filter of standard deviation 1 bin with half-sample symmetric reflection and truncate radius 4. Subtract the filtered minimum and divide by its sum times $\Delta$. These define $g$, not hidden reconstruction-dependent calibration. The RMS widths remain the acquisition-reported raw statistics; they are not recomputed from this differently processed profile.

## Method Hints

A maximum-entropy family multiplies a Gaussian prior by one-dimensional Lagrange functions evaluated along each transported direction. Relaxed ratio updates compare measured and predicted profiles. Four-dimensional sampling within finite cells can approximate those projections during reconstruction; deterministic uniform-cell integration can check them independently. Fit the covariance and construct the coordinate representation from the visible widths and optics rather than assuming a saved normalization matrix.

## References

1. A. Hoover. *Four-dimensional phase space tomography from one-dimensional measurements of a hadron beam*. Physical Review Accelerators and Beams 27,122802(2024). [doi:10.1103/PhysRevAccelBeams.27.122802](https://doi.org/10.1103/PhysRevAccelBeams.27.122802).
2. A. Hoover. *N-dimensional maximum-entropy tomography via particle sampling*. [arXiv:2409.17915](https://arxiv.org/abs/2409.17915). The conditional model and numerical output contract are fully defined above.
