---
name: oracle-differentiable-deflectometry
description: "Recover a refractive lens's two surface radii of curvature and center thickness from phase-shifted deflectometry fringe images captured by two calibrated cameras, by building a differentiable ray tracer and fitting lens parameters with Levenberg-Marquardt."
---

# oracle-differentiable-deflectometry Oracle Skill

This skill reconstructs the standard solution for a single-case refractive deflectometry inverse problem. The forward model projects camera pixels into 3D, refracts each ray through two aspheric lens surfaces via Snell's law (with implicit-differentiation Newton ray-surface intersection), and intersects the exit ray with a flat display screen. The measured target is produced by four-step phase-shifting fringe analysis, temporal unwrapping, DC removal, pixel-to-mm conversion, and reference-trace subtraction. A Levenberg-Marquardt solver with autograd column-wise Jacobians minimizes the per-pixel screen-intersection residual over 8 unknowns (two curvatures, thickness, 3D origin, two tilt angles). The final scalar outputs are surface_0_roc_mm=1/c0, surface_1_roc_mm=1/c1, and thickness_mm. Each operation below is independently removable so its contribution can be ablated; keep constants, axis orders, and conventions exactly as stated.

## Levenberg-Marquardt fit with autograd column-wise Jacobian

## Levenberg-Marquardt optimization

**Unknowns (8).** Parameter paths: `surfaces[0].c`, `surfaces[1].c`, `surfaces[1].d`, `origin` (3), `theta_x`, `theta_y`. Enable `requires_grad` on each.

**Initialization (critical).** `c0 = 0.001`, `c1 = 0.001` (NOT zero — zero gives a degenerate flat-surface Jacobian), `d1 = 3.0`, `theta_x = theta_y = 0.0`; `origin` from the mount estimate. Rebuild the pose (update) after setting.

**Residual and loss.** `residual = V[...,None] * (ps_cap - ps_pred)` with NaNs set to 0; loss `L = mean(residual^2)`.

**LM step (params: lambda0=0.1, mu=2.0, regularization='diag', max_iter=100):**
1. Predict `y = forward()`; record `L`. Stop if `L < 1e-16`.
2. Build the tall Jacobian `J` (M rows = valid residual entries, N=8 cols) column-wise via autograd JVP (perturb one parameter at a time using double-grad trick); reshape to `[-1, N]`. Form `JtJ = J^T J`.
3. Regularization matrix `R`: `'diag'` -> `diag(|diag(JtJ)|)` (Marquardt scaling for the very different parameter magnitudes, e.g. curvature ~1e-2 vs origin ~1e1); `'I'` -> identity.
4. Gradient term `b = J^T r` computed as `torch.autograd.grad(y, x, grad_outputs=residual)` per parameter, concatenated.
5. **Damping loop:** solve `(JtJ + lambda*R) delta = b` (`torch.linalg.solve`, fall back to `lstsq` on singular). Apply `+delta`. If new `L_current < L`: accept, `lambda /= mu`, break. Else revert (`-delta`), `lambda *= 2*mu`; abort inner loop after 20 tries or if `lambda > 1e16` or delta is NaN.
6. **Stopping:** after each outer iter, stop if `|L_prev - L| < 1e-8` or mean `|delta| < 1e-8`, or at max_iter. Typically converges in ~21 iterations.

Expected recovered values for this case: surface_0_roc ≈ -76.5 mm, surface_1_roc ≈ -31.7 mm, thickness ≈ 3.25 mm; mean displacement error ≈ 43 µm. Keep the diag regularization, the 0.001 curvature init, and the damping schedule.

Helper: `${CLAUDE_SKILL_DIR}/scripts/script_001.py`.
