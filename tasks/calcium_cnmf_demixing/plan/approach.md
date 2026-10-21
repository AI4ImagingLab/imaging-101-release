# Implemented mathematics and selected reconstruction

## Scope, observations and conventions

This is the native 3000-frame, 170-by-170 real Sue recording, not a planted synthetic movie. The complete selected release pipeline is NoRMCorre, 49 overlapping greedy/HALS patch initializations, rank-two background compression, correlation merging, global temporal fitting, two global spatial/temporal refits, classifier-free quality selection and diffusion-KDE relative fluorescence. Original numerical outputs are method references. No neuronal truth, spike labels or instrument-verified motion truth is asserted.

Index frames by `t=0,...,2999`, rows by `i`, columns by `j`, pixels in column-major order by `p=i+170*j`, source by `k` and background by `q=0,1`. The public movie has shape `(1,T,H,W)`; internal corrected movie has `(T,H,W)`; matrix `Y` has `(H*W,T)`. Internal spatial arrays `A` and `b` have `(H*W,K)` and `(H*W,2)`. Temporal arrays `C`, `S` and `f` have `(K,T)`, `(K,T)` and `(2,T)`. `A,b` are dimensionless; `Y,C,f` and innovations `S` use native arbitrary fluorescence counts. Relative fluorescence is dimensionless. Sampling is nominally 30 Hz, with 2 micrometre nominal pixel pitch and 0.4 second indicator decay configuration, not independent instrument calibrations.

## Subpixel motion estimation

Let `I` and `R` be an observed image and the current template; their transforms are `F(I)` and `F(R)`. The release registration maximizes the magnitude of their unnormalized cross correlation within the admitted displacement region:

```math
Q(\delta)=\mathcal F^{-1}\{\mathcal F(I)\overline{\mathcal F(R)}\}(\delta).
```

`F` denotes the discrete Fourier transform with the normalization used by OpenCV's DFT/inverse DFT calls. This is not phase-only correlation: dividing the cross spectrum by its magnitude would change the selected method. Locate the constrained integer maximum, then evaluate a localized matrix-multiply upsampled DFT on a tenth-pixel grid. `_upsampled_dft` explicitly evaluates the complex exponential kernel. `register_translation` returns the measured displacement; the image-content correction is its negative. Row and column are never swapped in the exported array.

Set additive bias `beta=-min(movie[:400])=296` from observations. Select every 61st frame (50 frames) for template initialization, using three original template alignment passes and median-of-10-frame-bin means. The small template alignment uses normalized correlation as a general correlation primitive; the main constrained Fourier estimator remains local mathematics. Template construction retains its original first-percentile offset rule if average intensity is negative. This numerical template rule does not clip or replace the public movie.

Rigid correction processes the full movie in 37 contiguous chunks. Global shifts lie within plus/minus 6 pixels. Average each corrected chunk, then take the median over 37 chunk templates. Piecewise correction uses that rigid template and nine 74-by-74 windows with starts `(0,50,96)` per axis. Each local displacement is restricted to plus/minus 3 pixels around the rigid solution. Interpolate the local shift field to the image with the release OpenCV resize/remap convention; perform cubic resampling with copied edge boundaries. Save corrected frames in float32 and add the observation-derived bias. Cubic interpolation can overshoot below zero despite the offset; preserving this behavior is required for parity.

Both the full rigid correction field `(T,2)` and the nine-window piecewise correction field `(T,9,2)` are exported. Window enumeration is row outer loop, column inner loop. No fresh reconstructed registered image is substituted with a cached image. Motion fixtures compare three complete nonrigid frames and all full-run fields.

## Pixel noise estimation

For each pixel trace `y` of length `T`, compute the one-sided periodogram using the original OpenCV real-input DFT branch. Select positive frequencies strictly above 0.25 and at most 0.5 cycles/frame. If `P` is the doubled one-sided periodogram, use

```math
\sigma=\sqrt{\operatorname{mean}_{\omega\in(0.25,0.5]} P(\omega)/2}.
```

The selected pipeline uses arithmetic mean, not the helper's default logarithmic mean. AR estimation also uses the selected arithmetic-mean noise estimator. The helper retains explicitly tested logarithmic-mean defaults for original standalone numerical cases; callers pass the selected convention. At 3000 frames the 3072 maximum FFT length never causes truncation.

## Overlapping patch initialization

Patch centers along an axis are `range(15,155,24)` followed by 155, yielding seven centers per axis. A patch spans center plus/minus 15 inclusive, clipped to the image, so final edge patches have width or height 30. All 49 patches use all 3000 frames. Four source components per patch give at most 196 initial source candidates. The selected patch background count is **one**, while global compression has rank **two**; conflating those parameters changes the method.

Normalize each patch by its temporal mean image plus the median of that image and float32 epsilon. Spatial and temporal initialization downsampling factors are both 2 using original area interpolation. These are internal initialization reductions; final observations, fitting and outputs retain native resolution and time. The Gaussian scale is `(4,4)` native pixels, support `(9,9)`, rescaled for initialization. The rolling activity sum window is 100 frames in the initializer. Greedily select four source centers from blurred residual energy. For a local patch matrix `V` and current temporal factor `c`, alternate five rank-one updates:

```math
a\leftarrow\max(Vc/(c^Tc),0),\qquad
c\leftarrow\max(a^TV/(a^Ta),0).
```

All variables here are local source factors and patch data; division and reshaping follow the extracted release code. Subtract each extracted rank-one contribution, initialize patch background by NNDSVDar NMF, then perform five full HALS sweeps over neuronal and background factors. HALS uses residual Gram updates, nonnegative projection and original spatial supports. Upsample both factors using the original interpolation order, restoring their scaling. Compute source-projected residual and normalize each spatial column to unit L2 norm while multiplying its temporal coefficients by that norm.

For concatenated factors `D=[A,b]`, temporal coefficients `Z=[C;f]`, and diagonal `N=diag(||D(:,k)||^2+eps)`, the projected residual is:

```math
R=\left[(D^TY)^TN^{-1}-Z^T(D^TD)N^{-1}\right]^T_{\mathrm{source}}.
```

The subscript means source rows only; it is not an image-domain residual. `eps` is float32 epsilon. The normalized source temporal trace plus `R` is used in subsequent merging/quality calculations.

## Stitching and background compression

Embed all patch factors into global pixel coordinates. Accumulate overlap count `m(p)` and divide each spatial source and patch background coefficient by `m(p)+eps`. Duplicate sources are retained until correlation merging; `del_duplicates=False`. Pixel noise uses the last patch assignment on an overlapping pixel, exactly as the release. Temporal source and projected-residual arrays are stitched as float32. No target-based mask or initializer is supplied.

Let `B` be the embedded overlap-normalized patch background matrix and `F` the 49 background traces. Fit nonnegative `F approximately W H`, rank two, by explicit NNDSVDar and projected coordinate descent. Randomized SVD uses oversampling 10, 7 LU power iterations when rank is below a tenth of the smaller dimension (otherwise 4), QR and reduced SVD. Flip singular vector signs deterministically. In the global fit, integer random seed 1 independently initializes the SVD random generator and the NNDSVDar zero-fill generator. This two-reset convention differs from a continuously advanced single generator and is covered by an independent full-global fixture. Patch NMF uses the inherited generator stream.

For one factor column `w`, with Gram `G=H H^T` and cross term `U=F H^T`, cyclic projected-coordinate updates use `w <- max(w-(W G(:,k)-U(:,k))/G(k,k),0)`. Update both factors each sweep. The projected-gradient violation is normalized by its first value; stop below `1e-10` or at 100 sweeps globally (patch initialization uses tolerance `1e-4`, maximum 200). No scikit-learn runtime is used.

Starting with `f=H`, perform exactly 100 background ALS cycles:

```math
f_q\leftarrow f_q/(\|f_q\|_2+\epsilon),\quad
b\leftarrow\max\{B F f^T(ff^T)^{-1},0\},\quad
f\leftarrow(b^Tb)^{-1}(B^Tb)^TF.
```

If a system is singular, use the documented least-squares fallback for the same linear subproblem. Temporal background coefficients are unconstrained in this ALS stage. Normalize each spatial background column; transfer its norm to the corresponding temporal row. Background factors have an additional basis ambiguity; evaluation compares their full product, not a particular factor array.

## Correlation-based rank-one source merging

Compute spatial overlap from positive off-diagonal `A^T A`; compute mean-subtracted Pearson temporal correlation only for overlapping candidates. Build connected groups using correlation greater than 0.85. Rank merge groups by summed pair correlations, descending. For each group, initialize spatial factor from component energy weights `sqrt(||a||^2 ||c+r||^2)`, then perform ten nonnegative rank-one updates on its source-plus-projected-residual movie. Normalize spatial factor and transfer scale to its trace. Deconvolve the resulting projected trace, retain its baseline-free calcium and residual, remove old grouped sources and append merged ones.

Repeat patch-stage merging until no groups remain. The bound upstream and local data both merge 37 groups, then one group, then none, leaving 149 sources. This number is an observed maintainer result, never a public target initializer. The refit merge stage has maximum 50 groups and finds none. The upstream fixed-AR merge branch passes an index where an AR coefficient is expected; that unexecuted branch is explicitly rejected by the local port instead of silently advertised as supported. It is not reached by this task's complete path.

## Global spatial refitting

For each source, dilate its support with a radius-two diamond footprint. Include background supports, normalize temporal source rows by their L2 norms and transfer norms into the spatial factors. For each pixel, select supported sources and background coefficients, and regress its full temporal observation against them.

The selected positive LARS regression uses standardized, mean-centered predictor columns and a centered response. If `X` is the selected temporal design and `y` is the pixel observation, its path is associated with:

```math
\frac{1}{2T}\|y-Xw\|_2^2+\alpha\|w\|_1,\qquad w\ge0,
\quad \alpha=\frac{\sigma_p\sqrt{\max_k\|C_k\|_2^2}}{2T}.
```

`sigma(p)` is the selected pixel noise; `k` ranges over supported neuronal rows; `T=3000`. Standardization has `with_mean=False` before the regression intercept centers the design. Returned coefficients follow the original standardized convention. Implement the equiangular positive LARS path explicitly, including active Gram solves, entry/drop events, float32 epsilon path stopping, maximum 500 path steps, and the original interpolation when the **remaining inactive** maximum correlation first crosses the requested alpha. This finite positive path is a release numerical convention and is not claimed to give exact convex KKT optimality for every small alpha. Ten independently captured real design matrices and a known orthogonal design verify the implemented path. Do not replace it with a generic optimum from a different solver.

Threshold each spatial component by a 3-by-3 median, cumulative squared-energy threshold 0.9999, 3-by-3 closing and strongest connected region. Remove empty components. Refit background shape with five nonnegative HALS sweeps. The full global memmap convention casts projected factors and accumulated blocks to float32; spatial and temporal projection use 19 pixel blocks to match the original streaming arithmetic. This is tested separately from nonstreamed patch cases.

## Temporal refitting and AR1 pool optimization

Compute projected residual using the same normalized Gram expression. Greedily color the overlap graph to obtain nonoverlapping source update groups. For each group, add its previous trace to its current projected residual and solve each one-dimensional temporal inverse problem. Subtract the induced Gram-weighted trace change from all projected residuals. Background temporal rows use an unconstrained residual update. Perform at most two temporal sweeps; stop early when Frobenius change is at most `0.001` times current temporal norm.

The initial global temporal fit uses `p=1`; the first global spatial refit uses `p=0` temporal nonnegative updates, followed by merge; the second spatial refit uses `p=1` deconvolution. In `p=0`, calcium is `max(y,0)` and there is no fitted AR model. In `p=1`, estimate the AR coefficient from autocovariances through lag 5, subtract noise variance at lag zero, solve the selected linear recurrence system and shrink stable AR roots by 0.96. Preserve the release clipping convention and its seed 45 when unstable roots require replacement.

For a trace `y`, baseline `d`, calcium `v`, noise `sigma` and innovation operator `Gv=(v(t)-g*v(t-1))`, the selected temporal problem is:

```math
\min_{v,d}\|Gv\|_1
\quad\mathrm{subject\ to}\quad
\|y-d-v\|_2^2\le T\sigma^2,\quad Gv\ge0.
```

The baseline is unconstrained in this task (`bas_nonneg=False`). The local pure-Python OASIS pool active-set method initializes one pool per sample. A pool has start, length, weighted value and squared-kernel weight; merge adjacent pools while the preceding predicted endpoint exceeds the next pool's value. Reconstruct each pool as a nonnegative exponentially decaying segment. Raise the dual sparsity multiplier using the explicit quadratic residual/noise-budget equation; for optimized baseline, update the mean residual and pool offsets. Stop optimized baseline iterations at relative noise-budget difference `1e-4`, vanishing calcium sum `1e-9`, or five baseline iterations. Some very small noise bounds are infeasible under nonnegative AR dynamics; the solver's maximum-iteration output is not advertised as a universally feasible exact optimum.

Use float32 pool weighted values and weights, trace arrays, noise and AR coefficient to match original compiled recurrence precision. `S(0)=0`; subsequent innovations are direct AR differences. These can be negative by about `5e-5` counts from float32 roundoff, retained with public absolute tolerance `1e-4`. `C` includes fitted baseline and initial exponential contribution. Subtracting and adding that initial contribution preserves the original rounding order. All final factors undergo spatial L2 gauge normalization with compensating temporal/residual/baseline/innovation scaling.

## Source quality and the investigated release discrepancy

The selected supported branch is `use_cnn=False`, fixed before oracle execution; no pretrained weights are used. Quality evaluates batches of 50 sources, preserving the release's within-batch neighbor convention. Subtract an interpolated 8th-percentile slow baseline using 600-frame bins. Estimate a trace mode by the released recursive half-sample heuristic; use values below the mode to estimate noise. For standardized fluorescence `z=(trace-mode)/(3*noise)`, sum `log(Phi(-z))` over 12-frame windows and take the most extreme window. `Phi` is the standard normal CDF. SNR is `-Phi^{-1}(exp(fitness/12))`.

For spatial quality, find up to ten positive activity peaks above the trace's min-plus-0.3-range threshold, collect frames from 5 before to 25 after each peak, and remove overlapping neighbors' active frames when possible. Compute Pearson correlation between the footprint and mean observed pixels over remaining active frames. This selected release implementation does not explicitly subtract the background in that correlation despite broader prose in the upstream docstring; local code preserves executed mathematics.

Accept when `(SNR>2 or spatial_r>=0.85) and SNR>0.5 and spatial_r>-1`. Both oracle and fresh result accept 102 of 149 sources. A source is not a verified neuron merely because it passes these thresholds.

Independent inspection found the released three-point mode branch repeats the left-narrower condition, returning the middle point when the right interval is narrower. This is preserved under the explicit name `release_compatibility_mode`, **not** described as a paper equation or mathematically correct symmetric estimator. Independently captured active three-point cases contain 135 affected configurations among 298 calls. `symmetric_half_sample_mode` provides a separate diagnostic corrected definition. On the fresh complete result, corrected and released modes both select the same 102 components; maximum SNR change is 0.513195 and median absolute change 0.000858851. Notebook comparisons show both. This disclosed quality compatibility discrepancy does not alter the forward model, CNMF factors or OASIS inverse.

The released even-length recursion also iterates only the first `ceil(n/2)` start positions, omitting the final admissible half interval when `n` is even. The independent symmetric estimator enumerates every admissible start. For `[0,10,20,20.1]`, the released estimator returns 5 while the symmetric shortest-half estimator returns 20.05. The reported corrected-mode SNR comparison includes both this boundary correction and the three-point correction; it must not be attributed solely to the three-point branch.

## Relative fluorescence and evaluation

Let `F=normalized(C+R)` and `B=A^T b f` be source-projected signal and background. Estimate each source's baseline percentile from its first 250 samples using diffusion KDE: histogram on 4096 bins over observed range extended by 10%, DCT smoothing, Botev fixed-point bandwidth root, inverse DCT normalized density and CDF at the density mode. If bandwidth finding fails, repeat the sample list up to ten times; invalid percentile falls back to the median, exactly as release. Apply the chosen percentile to both `F` and `B` with 250-frame nearest-reflection percentile filtering. Relative fluorescence is `(F-Fd)/(Bd+Fd)`, where `Fd,Bd` are these two baselines. Quantile argument 8 is retained in original API context, but automatic selection overrides it in the shipped path.

The Judge uses one independently executed method reference. It matches source spatial columns by fixed Hungarian assignment, rejects pairs below 0.7 cosine, penalizes unmatched sources with detection F1, and evaluates every required joint output. Spatial gauge is unit L2; no post-fit intensity scaling, shifts, crops or magnitude conversions are allowed. Background is compared through its full movie, since factor basis is nonidentifiable. Relative and absolute tolerances, precision evidence, independent repeat, fixtures and corrected-mode analysis are indexed in `test/validation.md`. Judge agreement cannot establish physiological accuracy.
