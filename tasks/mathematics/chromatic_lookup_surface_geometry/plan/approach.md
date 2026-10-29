# Implemented empirical surface inverse

## Unknowns and acquisition

The unknown z has shape(3,650,650), units mm, and is camera-axial depth on three chronological measured exposures0,1,2. Camera rows point down and columns right. Public raw pattern/white data are uint16 ADU, cropped without interpolation from native sensor coordinates(left,top,right,bottom)=(650,200,1300,850). There is no ambient capture, target geometry or target-derived calibration. The independently measured planar-board factors represent three RGB dictionaries and an axial-depth dictionary at749 ordered samples. Rank25 is a supplied instrument representation rather than a solver hyperparameter.

Let f be frame, p the row-major flattened pixel, c RGB channel, k the ordered calibration sample and r the low-rank coordinate. Define minimum white brightness m and support M, normalized RGB q, color dictionary C and depth dictionary D:

```math
m_{f,p}=\min_c I^W_{f,p,c},\qquad
M_{f,p}=\left[m_{f,p}>0.05\max_j m_{f,j}\right],\qquad
q_{f,p,c}=I^P_{f,p,c}/I^W_{f,p,c},
```

```math
C_{p,k,c}=\sum_{r=1}^{25}L^C_{c,p,r}R^C_{c,r,k},\qquad
D_{p,k}=\sum_{r=1}^{25}L^D_{p,r}R^D_{r,k},\qquad
\mathbf q_{f,p}=\mathbf C_{p,k_{f,p}}+\boldsymbol\eta_{f,p}.
```

I are detector intensities, m is ADU, M is Boolean, q and C are dimensionless intensity ratios, D and z are mm, and eta represents measurement/model discrepancy. Color left factors have ratio units, depth left factors mm, and all right factors dimensionless. The gauge of these supplied factor products is fixed; no independent factor rescaling is needed. The sample-depth trajectory is ray-dependent; planned0.2mm stage translation does not imply uniform axial-depth spacing. No boundary continuation is assumed outside these visible independent camera rays.

Normalization uses uint16 division in float64 assigned to float32. Unsupported q is zero internally. Original safe nan/overflow handling and nonnegative clipping are preserved; all supported white channels are strictly positive by the input-only rule. No residual or reconstructed depth changes support.

## Primary exhaustive inverse

For each supported pixel, minimize the finite unweighted RGB squared distance and read its associated depth:

```math
E_{f,p}(k)=\sum_{c=1}^{3}(C_{p,k,c}-q_{f,p,c})^2,\qquad
\widehat k_{f,p}=\arg\min_{0\leq k<749}E_{f,p}(k),\qquad
\widehat z_{f,p}=D_{p,\widehat k_{f,p}}.
```

An exact tie selects the earliest sample. The three squared terms accumulate sequentially in float32, as in the source; the returned internal loss is the Euclidean RGB norm, not squared distance or a per-channel RMS. The factor matrix products remain float32. There are no objective weights, fitted gains, regularizers, iterative initialization, gradient, step size, line search or convergence tolerance in this finite exhaustive inverse. Stopping means all749 entries have been compared for every supported ray.

Process the union of the three observation masks in blocks of2048 rays. Expand RGB/depth factor products only for that block, then decode each frame using its own support. This preserves the original finite dictionary while avoiding a4.7GiB full expansion. Unsupported submitted depth is exactly−1mm. Internal full-image index is−1 and loss infinity there; these internal arrays are never submitted or scored as geometry. The low-level3D ray API uses source index0 outside support; solve explicitly preserves the original4D image API's−1 convention.

## Actual optional helper comparisons

The notebook actively computes the source's fixed-window coarse-to-fine helper at strides(8,4,2,1), half-windows(374,125,50,25), epsilon0.1. The start is clipped to[0,749−2delta], cast uint16; exactly2delta candidates are searched. Hence the initial748-entry window excludes the last sample. Intermediate index/prior arrays use int16, nearest-neighbor fill for nonpositive index and loss above epsilon, followed by a Gaussian filter with nearest boundary and sigma equal to the stride ratio; the integer output behavior is preserved. The final helper returns a window-relative index. These are source conventions, not a proposed improved adaptive protocol.

Temporal continuation uses the preceding exhaustive **global** index and half-window25 on the two adjacent transitions. It searches the same clipped window, then adds its start to recover a global index. It has no additional fallback. Source CLI GPU-argument/missing-epsilon/prior issues are not called or silently repaired. These controls are not the paper's complete adaptive widening or a physical-accuracy ranking.

The source's depth-gather broadcasting error in the helper was fixed only in a private oracle copy, after an independently known4x5x9 gather case. Primary source CPU scalar sqrt on an image was similarly qualified to array sqrt after known norm/tie checks. General disk allocation changes preserve full-shape source matrix multiplication and exact copied bytes. All qualifications, original failures and diffs are private and identified in validation; no unchanged-source claim is made.

## Review experiments and visualization geometry

Compression controls truncate the first5/10/15/20 factors and compare with full25 on allthree observations. This changes both dictionaries and is not a new calibrated instrument. Ratio-noise controls add zero-mean Gaussian draws sigma0.002/0.005/0.01, seed232323, retaining negative draws and fixed support. They are numerical sensitivities, not measured camera uncertainty. Supplementary real frames212/425/637/849 are actively reconstructed with the same original calibration and support; their two nearzero output failures remain in every quantitative comparison. The initial three-frame binding predates all inverses; extra cases were not used to select an easier benchmark.

An independent planar-motion trajectory diagnostic fits depth versus ordered index excluding510 on four fixed rays. Approximately affine remaining748 samples make the nearzero entries inconsistent with this hypothesis. It neither establishes a cause nor authorizes replacing calibration or masking outputs. The full original observations/fields remain unchanged.

The optional XYZ helper uses dataset intrinsics fx3580.915139167154, fy3578.1633141223797, cx967.7411991347725, cy509.395929381727 pixel, five Brown–Conrady undistortion iterations with(k1,k2,p1,p2,k3)=(−0.05202839890657609,−0.05977211653771139,0.0014034145040185428,−0.00031193552659360146,0.3585615042719389), and the fixed ROI offset. Normalized undistorted ray(x,y,1) maps to(X,Y,Z)=(xz,yz,z) mm. This is visualization only; no geometry parameters enter the primary empirical depth inverse. Independent pinhole and nonzero-distortion roundtrip tests validate the helper, not acquisition accuracy.

## Evaluation conventions

Each complete frame mask is scored independently: uncentered cosine NCC at least0.999999, RMS depth error divided by that reference's dynamic range at most1e−6, and absolute maximum depth error at most0.001mm. Source repeats are bitwise; nonzero tolerances cover numeric float32 differences, not scientific uncertainty. Truth takes priority if a valid truth asset exists; this task has only a qualified method reference.

An independent forward evaluator chooses the nearest calibrated axial depth to each submitted depth, earliest at ties, evaluates its supplied RGB factors in float64, and computes the RMS Euclidean RGB residual over all supported pixels. Per-frame boundaries are1.1 times independently measured original RMS:0.07743929173549902,0.07735535361821792,0.07748206427663368. These were fixed before evaluating the first full cleaned output. No registration, scale, crop, residual mask or error clipping is allowed. A lower photometric residual alone does not establish correct geometry; calibration ambiguity and two supplementary failures explicitly demonstrate this.
