# Solution Approach

## Algorithm: ADMM with Nonconvex Sparsity Penalty

### Overview

We solve the phase unwrapping problem by finding the phase field Phi whose gradient differs from the wrapped gradient phi at the fewest possible pixels. This is an L0 minimization problem, which we relax using the G0 penalty and solve via the Alternating Direction Method of Multipliers (ADMM).

### Pipeline Stages

1. **Preprocessing**: Load complex interferogram, extract wrapped phase and coherence, compute wrapped gradients with the standard adjustment (wrapping finite differences to [-pi, pi]).

2. **Forward model**: The discrete gradient operator D maps a phase field to its x and y gradients. The adjoint D^T (divergence) and the Laplacian D^T*D are used in the linear solver. The Laplacian is diagonalized by the DCT under Neumann boundary conditions, enabling O(n log n) solves.

3. **ADMM solver**: At each iteration:
   - **Linear solve**: Solve (D^T D) Phi = D^T(w + phi - Lambda) via DCT
   - **Shrinkage**: Apply G0 shrinkage to (D*Phi - phi + Lambda)
   - **Multiplier update**: Lambda += c * (D*Phi - phi - w)

4. **Post-processing**: Optionally snap to congruence (unwrapped = wrapped + 2*pi*k).

5. **Evaluation**: Compare against SNAPHU reference using RMSE, fraction of pixels agreeing within pi, and visual inspection.

### Key Design Decisions

- **Neumann boundary conditions**: Natural for bounded image domains; zero gradient at edges.
- **DCT-based solver**: O(n log n) vs O(n^2) for direct sparse solvers. Dominates per-iteration cost.
- **G0 penalty (p=0)**: More aggressive sparsity promotion than L1, giving better unwrapping at phase discontinuities.
- **Convergence criterion**: Max pixel change < pi/5 radians (about 36 degrees).

### Expected Results

- Convergence in ~15 iterations on the test data (778x947 pixels)
- Runtime under 1 second on modern hardware
- >99.7% of pixels agree with SNAPHU within pi radians
- All pixels agree within 2*pi radians

## Default Parameters

These values are hard-coded in `main.py` and were removed from `meta_data.json` (which is reserved for imaging parameters only):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_iters` | 500 | Maximum ADMM iterations |
| `tol` | π/5 ≈ 0.6283 | Convergence tolerance (radians) |
| `lmbda` | 1 | ADMM regularization parameter λ |
| `p` | 0 | Sparsity penalty exponent (0 = G0 shrinkage) |
| `c` | 1.3 | ADMM penalty parameter update factor |
| `dtype` | "float32" | Computation dtype |
