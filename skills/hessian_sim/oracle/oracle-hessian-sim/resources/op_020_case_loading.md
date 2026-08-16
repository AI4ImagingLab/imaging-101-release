# Case loading constants

## Raw layout
`raw_frames` shape `(1, 72, 256, 256)`; take index [0] -> `(72, 256, 256)` float64.
Frame index `f = t*6 + a*3 + p` for time `t` in 0..11, angle `a` in 0..1, phase `p` in 0..2.
Per time point order: `[a0p0, a0p1, a0p2, a1p0, a1p1, a1p2]`.

## Grid
- sy = sx = 256
- n = max(sy, sx, 512) = 512 (single-band padded grid)
- 2*n = 1024 (extended super-resolution grid)
- output crop is centred 512x512 out of the 1024 grid

## Fixed constants (must match exactly)
| name | value | role |
|---|---|---|
| regul | 2*pi | full pattern phase span |
| spjg | [4,4,3] | integer phase-step ratios (from params) |
| beishu_an | 1 | calibration frame-average factor |
| beishu_re | 1 | reconstruction frame-average factor |
| pg | ceil(512*2*NA*px/lambda * n/512) ~ 123 | expected pattern radius (2n px) |
| fanwei | ceil(50 * n/512) ~ 50 | search annulus half-width |
| weilac | 2.0 | Wiener regularisation strength |
| mu | 150.0 | Hessian data-fidelity weight (siranu) |
| sigma_z | 1.0 | Hessian z-axis weight (zbei) |
| n_iter | 100 | Hessian split-Bregman iterations |

## Calibration subset
`est_frames = raw[: nangles*nphases*beishu_an]` = `raw[:6]`.
