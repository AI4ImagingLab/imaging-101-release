# Implemented geometry and discrete inverse

## Binding and coordinates

This is the complete real Adirondack Q pair, 496 rows by 718 columns, with native RGB counts. It reproduces the selected BMVC2015 two-predecessor MGM8 method, not all 37 full-resolution paper examples. Let u,v denote integer column,row coordinates. In the left-camera frame X points right, Y down and Z forward, all in mm. Rectified focal length f=1038.018 pixels, baseline B=176.252 mm, principal points (322.037,243.393) and (375.308,243.393), and a=53.271 pixels. Projection and inverse are

```math
u_L=fX/Z+c_L,\quad u_R=f(X-B)/Z+c_R,\quad
D=u_L-u_R,\quad Z=fB/(D+a).
```

The source's native integer label d=uR-uL=-D is searched in ascending order -127,...,0. This fixed 128-label choice was bound before local implementation; target-specific author ndisp/vmin/vmax are not released as calibration. No rectification correction, vertical search, median filter, subpixel refinement, left-right rejection, adaptive support or target-derived calibration is used. Photometric agreement is approximate because of real occlusion and reflectance.

## Executable unary normalization

For each RGB channel, enumerate offsets dy=-2..2 then dx=-2..2, omitting (0,0), and record the strict bit I(center)<I(neighbor). Channels precede spatial offsets. Unavailable neighbors compare false, including on a constant field. Pack 72 bits big-bit-first into 9 bytes and append 3 zero bytes, matching the source's three-float-word storage. Matching uses total Hamming bit count **divided by three packed words**, double division then float32 storage. The source CLI separately multiplies P1=8,P2=32 by the original three image channels, so effective penalties are **24,96** in these actual unary units. Both conventions are preserved; these are not raw summed Hamming costs with unscaled paper penalties. The first failed local extraction exposed this exact normalization error against independent fixtures and was corrected without changing any criteria.

C(p,d) is the resulting nonnegative unary cost at left pixel p and right pixel p+(d,0); invalid right coordinates cost positive infinity. If all labels are invalid the source fills the pixel's vector with zero, though the bound interval always includes the in-frame d=0. There is no likelihood scale or calibrated noise variance implied by this rank cost.

## Objective family and source recurrence

The discrete pairwise energy has unary C plus a neighborhood regularizer V(d,e): zero for d=e, P1 for absolute difference one, and P2 otherwise. MGM approximately aggregates directional messages; it does not exactly minimize the two-dimensional energy. For a predecessor vector q define

```math
m(q,d)=\min\big(q(d),\min(q(d-1),q(d+1))+P_1,\min_e q(e)+P_2\big)-\min_e q(e).
```

Here q(d) is accumulated predecessor cost, e ranges over native labels, and absent label neighbors cost infinity. In a two-predecessor pass, L(p,d)=C(p,d)+m(q,d)/2+m(r,d)/2. The implementation computes the two halves, adds them, then adds C, in float32 source order. Each pass initializes every L vector to C. Updates occur only if **all four source-associated neighbors** are in the image, including the two unused neighbors. Consequently boundary vectors can remain unary. This source boundary rule is preserved explicitly.

Eight source pass configurations are stored as q,r,s,t offsets in (x,y), increasing-x/increasing-y flags and row-major flag in `solvers.PASSES`. They cover opposite axial and diagonal orientations in the source order. Traversal uses x=ii-2*jj,y=jj, swaps axes for column-major, then applies reversals. Every used predecessor is in an earlier wave. Vectorizing the independent pixels in a wave changes neither causality nor update order at a pixel.

Sum the eight L vectors in pass order. Apply overcount correction S=sum(L)-(8-1)C. Invalid labels produce inf-inf NaN, exactly as in the source, and are excluded from selection. Select the first strict minimum over ascending native labels; ties therefore choose the largest positive D, not the smallest. A single traversal of each direction is the stopping rule; no artificial outer iteration, tolerance, optimizer or training is introduced. There is no RNG. Return D=-d as float32 and evaluate calibrated Z in float64 before float32 serialization.

## Active comparisons and parameter effects

WTA selects the unary alone. SGM uses only q and does not subtract repeated unary; ocSGM uses q and applies the correction; MGM uses q,r and correction. The unchanged original CLI was separately run for SGM8, ocSGM8, MGM4, MGM8 with CLI penalties4/16 and16/64. Fresh local comparisons use exactly these parameters. The primary is MGM8 with8/32. Four-direction or penalty comparisons are diagnostics, not substituted primary methods or altered acquisition inputs. Every plotted result is freshly computed by local modules.

## Evaluation, physical checks and limitations

Independent structured-light measurements supply actual disparity on354918 pixels, including occlusions;1210 unavailable pixels retain their native sentinel values in hidden assets. Hidden reference depth is calibrated from those measured disparities, not a reconstruction. The fixed availability mask does not depend on reconstruction errors. Output arrays must be finite everywhere; both quantities are scored, without crop, scale fit or alignment. NCC is flattened cosine; NRMSE is RMS error divided by reference dynamic range. Disparity bad1 is the fraction with absolute error greater than1pixel. Additional maximum relative depth/disparity consistency is over the entire output grid, limited to3 float32 epsilons. Baseline/threshold numeric values are in evaluation/metrics.json, frozen before extraction from independent original runs and known poor controls. NCC boundaries are0.9 times original NCC, NRMSE and bad1 limits1.1 times original errors. No threshold was relaxed after local results.

Frozen source comparisons check complete Census bytes, selected unary vectors, all eight recursive passes and aggregated costs, plus the full disparity. Costs allow absolute2e-4, disparity1e-5pixels; Census/unary equality is exact. Independent tests project known3D points, exhaustively minimize a small label potential, check monotone-radiometry invariance and recover a known integer image shift. Geometric unit/sign checks do not rely on source agreement. True-reference quality is limited by occlusion/weak texture, the approximate method and finite structured-light accuracy; native large errors remain visible and scored.
