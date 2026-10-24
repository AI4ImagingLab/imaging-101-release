# Implemented mathematics

## Physical unknown, units and acquisition

The standard bound input is the complete400x600 author-provided real RGB photograph, normalized from uint8 by255 and stored as float64. Gamma=1 is the source camera approximation, not verified linear photon flux. Airlight A=(.72,.785,.81) is an observation-based estimate supplied in the original parameter file; it is not inferred from a hidden clear view. The physical equation is I=tJ+(1-t)A. Here I,J,A have normalized-intensity units and t is dimensionless. Pixel rows point down and columns right. Absolute distance and atmospheric attenuation are not separately identified. This task evaluates the source's restored display plus inferred spatial transmission, not a calibrated radiation or range map.

## Spherical prior and discretization

Subtract A from each RGB pixel and compute radius r=sqrt(sum_c(I_c-A_c)^2), where c indexes three color channels. The direction is (I-A)/r. No benchmark pixel has r=0; the local function explicitly rejects this unsupported degenerate input rather than silently inventing a bin. The1000points in SPHERE_POINTS are the exact parsed4-decimal source TR1000.txt entries. They are method discretization, not public instrument calibration. Do not renormalize those approximate unit vectors. scipy.spatial.cKDTree is a general nearest-point primitive; source Octave oracle's independent blocked Euclidean implementation verifies the same integer bins on every benchmark pixel.

For each bin k compute maximum radius R_k, population n_k and sample radial standard deviation s_k, with singleton std=0. At each pixel p in bin k: t0(p)=clip(r(p)/R_k,.1,1). Impose lower bound1-min_c(I_c/A_c) on this initial field. Define s(p)=s_k and reliability v(p)=min(1,n_k/50)*min(1,3*max(.001,s(p)/max(s)-.1)). Normalization uses global max(s), not per-bin maxima. The image **before** radiometric gamma inversion guides the source WLS stage, matching the original source. Benchmark gamma1 makes the distinction numerically irrelevant here.

## Weighted graph refinement

Compute G=.298936*R+.587043*Gchannel+.114021*B using the Octave8.4 verified reference conversion. These exact coefficients are an identified modern oracle convention, not a claim of bitwise identity with every MATLAB release. For every horizontal or vertical neighbor pair(p,q), define c_pq=lambda/((G_p-G_q)^2+epsilon), with lambda=.1,epsilon=1e-5. No periodic or exterior edges exist.

Subtract min(v) from v and divide by max(v-min(v))+1e-5. Call the result w. At top-row columns where w<.6, replace w by.8 and replace t0 by the minimum initial transmission of that column. Call the modified initial field a. This is an explicit source boundary anchor, not an independent depth measurement. Let L be the symmetric positive-semidefinite weighted graph Laplacian. Minimize

```math
Q(t)=\sum_p w_p(t_p-a_p)^2+\sum_{(p,q)\in E}c_{pq}(t_p-t_q)^2.
```

E lists each unordered neighbor pair once. No1/2 or division by image size is included. Its gradient is2[(diag(w)+L)t-diag(w)a]. The direct sparse normal equation is (diag(w)+L)t=diag(w)a. Column-major flattening is preserved in the diagonal assembly; no iterative initialization, step size, line search or stopping criterion applies. scipy.sparse.linalg.spsolve supplies only a general linear-system solve. An independently enumerated graph-energy test, symmetry, constant-nullspace and positive-eigenvalue tests verify the mathematics beyond source copying. The benchmark residual is also checked in the complete normal equations. Nonnegative connected graph weights and positive top anchors yield a unique bounded weighted-average solution for this acquisition.

The source uint8 demo is separately run and archived. Crucially, non_local_dehazing.m calls im2double before passing guidance to WLS, so both the uint8 input path and the normalized float64 input path use floating-point guidance and produce identical results. An initial concern based on inspecting isolated integer grayscale arithmetic was disproved by tracing this actual caller path. It is not an upstream defect. The two representations provide an independent preprocessing-equivalence check, not two different methods. The declared public observation stores the exact normalized values consumed by the selected original run. All graph normal-equation checks use the explicit quadratic above.

## Radiance and display conventions

The selected source reconstructs R0=(I-(1-1.06*t)A)/max(t,.1), clips componentwise into[0,1], and applies power1/gamma. The residual-haze factor1.06 adds .06*A for t>=.1 before clipping; therefore this is deliberately different from unbiased J=(I-(1-t)A)/t. Preserve it; do not silently relabel R0 as exact physical radiance.

The display stage subtracts a **global** RGB minimum, divides by the shifted global RGB maximum, and computes per-channel order statistics. For N=240000 pixels the lower0-based index is floor(.005N), and upper index is ceil(.995N)-1, without interpolation. Equal lower/upper bounds become[0,1]. Floor each upper bound at.2; mix upper b by .2b+.8max(b,mean(b)). Mix lower a by .2a+.8min(a,mean(a)). Means are across RGB and taken before the corresponding simultaneous adjustment. Linearly remap each channel between its modified limits, clip into[0,1], multiply by255 and round positive half-up to uint8. The local restored image matches the normalized-input original byte-for-byte on this acquisition. Output transmission remains float64, with no display clipping or rescaling.

## Notebook-only scientific comparisons

Actively reconstruct the complete benchmark, evaluate the fresh submission and compare to independent original measurements. Compute a no-refinement ablation from actual t0; independently solve lambda=.03,.1,.3; report exact weighted data and smoothness contributions, row220profiles and differences to the fixed standard. No comparison is filled by zeros. Frozen original uint8-input output supplies an explicitly labeled input-representation equivalence check. The physical inverse with leave_haze1 and the source leave_haze1.06 illustrate display effects; forward the clipped physical inverse through I=tJ+(1-t)A to show where clipping changes consistency. No claim of physical accuracy or depth truth follows from a small residual. Every figure is derived from bound data and actual computed arrays.

## Evaluation conventions

The reference is independently executed original normalized-input output, not physical truth. Each output is scored without gain, translation, masks, cropping or color fitting. NCC is flattened cosine similarity (zero submitted norm gives0); NRMSE is RMS difference divided by the reference dynamic range. Each required output must satisfy both boundaries. Nonzero method-consistency tolerances are specified in metrics.json: source/local transmission maxabs1.96e-13 and exact RGB agreement motivate float64 sparse-solver and integer-rounding envelopes. The test predeclared stage tolerance is rtol1e-9,atol1e-11; integer bins and the chosen benchmark's restored RGB use exact equality. Strict errors reject wrong keys/shapes/dtypes/nonfinite values or unavailable evaluator assets. Valid poor images fail, including zero display images. Judge imports no solver.
