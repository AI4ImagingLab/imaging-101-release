# Cell-Membrane FRAP Transport

> Recover a diffusion coefficient and a space–time fluorescence recovery field from a complete measured photobleaching movie whose initial bleach profile, camera noise and ongoing bleaching obscure transport.
> Domain: biology | Keywords: fluorescence microscopy, molecular transport, inverse problems | Difficulty: Hard

## Background

Fluorescence recovery after photobleaching measures molecular mobility by deliberately extinguishing fluorescent labels in a small region and observing the subsequent return of fluorescence. Motion carries unbleached molecules into the affected region. The resulting space–time signal contains information about transport even when the original bleach pattern is irregular and the fluorophore concentration cannot be independently measured.

Membrane transport controls how receptors, transporters and other proteins redistribute within cells. Recovery microscopy can distinguish slow lateral diffusion from exchange with another molecular population. Interpretation is difficult because imaging also bleaches labels, optical blur removes fine spatial detail and a camera background contributes intensity without participating in transport.

## Problem Description

The supplied acquisition is a complete 42-frame cell-membrane FRAP movie, accompanied by two observation-defined sampling paths. One follows the bleached membrane region and the other an unbleached control region. Infer the effective lateral diffusion coefficient and the optically resolved recovery field along the first path. The file-specific organism, labeled protein and microscope model are not independently identified in the distributed acquisition metadata; no such identity is required by this inverse problem.

Let $x$ denote membrane arc length, $t$ time after bleaching, $D$ the diffusion coefficient, and $c(x,t)$ a fluorescence-proportional concentration field. The acquisition model is

```math
\partial_t c=D\partial_x^2c,\qquad
I(j,n)=B+e^{-\epsilon n}\int_{\mathbb R}h_\mu(x_j-u)c(u,n\Delta t)\,du+\eta(j,n).
```

Here $I$ is sampled camera intensity, $B$ the constant camera background, $\epsilon$ the imaging-induced loss per frame, $n$ the postbleach frame index, $\Delta t$ the frame interval, and $\eta$ uncalibrated measurement noise. The optical response $h_\mu$ is a Gaussian of standard deviation $\mu$; the standard method uses the nominal one-pixel response convention below. Its associated diffusion kernel is Gaussian with variance $2Dt$. Neither the unknown initial profile $c(x,0)$ nor boundary values outside the observed line are provided. The observed window is part of an infinite-line approximation, not a periodic domain. Absolute molecular concentration cannot be obtained without a fluorescence-to-concentration calibration.

The requested field is the noise-reduced optical intensity at the observed times, including the measured background and imaging loss, represented in the nine real spatial Fourier modes specified below. It is **not** a claim of subpixel molecular concentration recovery. Infer $D$ jointly with the unknown optical profile; reporting an arbitrary smoothed movie without transport inference does not solve the task.

Save `output/reconstruction.npz` relative to the task workspace, with both required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 41, 89) | float64 | Real fitted optical intensity, original camera a.u.; axes batch, postbleach time, arc-length position; includes background and imaging loss. |
| diffusivity | (1, 1) | float64 | Real positive effective lateral diffusion coefficient, µm²/s; axes batch, parameter. |

All values must be finite. Preserve the camera intensity scale and increasing frame/path order. Postbleach time zero is movie frame index 1; the prebleach frame index 0 informs the acquisition context. Position zero is the first supplied FRAP path point. No fitted rescaling, background subtraction, crop, shift, phase adjustment or temporal alignment is applied by evaluation.

The complete recovery field is evaluated with cosine similarity (NCC) and dynamic-range-normalized root mean square error (NRMSE). For flattened estimate $a$ and comparison field $b$, NCC is $a^Tb/(\lVert a\rVert\lVert b\rVert)$, with zero assigned to an all-zero estimate, and NRMSE is $\sqrt{\mathrm{mean}((a-b)^2)}/(\max b-\min b)$. The scalar diffusion metric is $|D/D_\mathrm{comparison}-1|$. Every criterion must pass. These metrics measure reproducibility of the standard inference under the declared model; they do not establish independently known biological truth.

If $G$ maps an initial profile to compressed observations, $K=GG^T$, $y$ is the compressed observation vector after subtracting known offsets, and $\lambda=\sigma_{\max}(K)\,369\,\epsilon_{64}$ is a fixed positive numerical regularizer, with largest singular value $\sigma_{\max}$, 369=41 times nine compressed samples, and float64 machine epsilon $\epsilon_{64}=2^{-52}$, the residual projector is $P=\lambda(K+\lambda I)^{-1}$; $I$ here is the identity matrix. Minimize $\lVert Py\rVert^2/\mathrm{tr}(P^TP)$ to account for the effective residual degrees of freedom. The fitted observation is the original observation minus the unnormalized projected residual. This formulation avoids assuming that the noisy first postbleach frame is an exact initial condition. Near-null modes of the initial field are not identifiable and should not be interpreted as resolved biological structure.

The diffusion search is restricted to [(89/9)^2/(41*40), 89^2/5] in pixel-squared/frame units; convert to micrometre-squared/second with the squared pixel size divided by frame duration. This interval defines the conditional diffusion prior, without fixing an optimizer trajectory.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`. It contains the complete movie and observation-defined sampling geometry, without a precomputed recovery field.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| movie | (1, 42, 519, 408) | uint16 | Native observed intensities, camera a.u.; axes batch, acquisition time, image row y, image column x. |
| frap_coordinates | (1, 89, 2) | float32 | Bleached-region centerline coordinates in image pixels, ordered x then y; axes batch, arc-length sample, coordinate component. |
| control_coordinates | (1, 75, 2) | float32 | Unbleached control centerline coordinates in image pixels, ordered x then y; axes batch, arc-length sample, coordinate component. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Pixel size | 0.0431 | µm/pixel | Author acquisition-setting value; computations retain its float32 representation |
| Frame interval | 30 | s | Author file-specific setting; all 42 frames retained |
| Prebleach frames | 1 | frame | Movie index 0 |
| Recovery frames | 41 | frame | Movie indices 1–41; postbleach times 0–1200 s |
| ROI strip width | 9 | pixel | Transverse offsets −4 through +4 about each supplied centerline |
| Coordinate origin | Upper-left pixel center is (0,0) | pixel | x increases rightward; y increases downward |
| Path normal | (−dy,dx) normalized from current minus previous point | dimensionless | The first predecessor is extrapolated from the first two coordinates; normal direction is immaterial for symmetric sampling |
| Sampling | Bilinear intensity interpolation; float32 intermediate strips | camera a.u. | All strip pixels lie inside the observed image |
| Strip reduction | Triangular 9-to-1 antialiasing with replicated edge samples | dimensionless | Kernel weight at source coordinate s is 1−abs((s+0.5)/9−0.5), s=−4,…,12, clipped to strip indices 0,…,8 and normalized |
| Background rectangle | x in [63,204), y in [50,452), all frames | pixel | Sample-free region; background estimated from visible observations |
| Optical response | Gaussian; nominal µ equals one pixel | pixel | Method approximation, not independently measured instrument calibration |
| Noise | No independently calibrated distribution or variance | camera a.u. | Equal-weight inference; no calibrated chi-square claim |
| Spatial representation | DC; cosine frequencies 1,2,3,4; sine frequencies 4,3,2,1 over 89 samples | cycles/window | Orthonormal real Fourier basis; limits the evaluated optical field bandwidth |
| Transport boundary | Infinite-line approximation | — | No prescribed unobserved boundary profile; no periodic transport assumption |
| Absolute fluorescence calibration | Unavailable | — | Output remains camera intensity, not molecules/µm |

The distributed movie is an author-supplied processed experimental file. The file-specific 30 s interval is authoritative here; it must not be replaced with timing reported for a different experimental series.

## Method Hints

A variable-projection approach can infer transport while treating the initial bleach profile as an unknown linear field. Compress observed spatial profiles into a small orthonormal Fourier basis, form the sampled heat-kernel Gram matrix, and eliminate the initial field by regularized linear projection. Minimize the remaining residual over the diffusion coefficient. The unbleached control constrains imaging-induced loss separately from diffusion.



## References

E. Lorenzetti, C. Municio-Diaz, N. Minc, A. Boudaoud and A. Fruleux, “Inferring diffusion, reaction, and exchange parameters from imperfect FRAP,” *Biophysical Journal* 124, 2941–2960 (2025). [DOI: 10.1016/j.bpj.2025.07.036](https://doi.org/10.1016/j.bpj.2025.07.036).

M. A. Branch, T. F. Coleman and Y. Li, “A subspace, interior, and conjugate gradient method for large-scale bound-constrained minimization problems,” *SIAM Journal on Scientific Computing* 21, 1–23 (1999). [DOI: 10.1137/S1064827595289108](https://doi.org/10.1137/S1064827595289108).
