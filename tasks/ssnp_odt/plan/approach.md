# SSNP-ODT: Solution Approach

## Problem

Recover a 3D refractive index contrast volume Δn ∈ ℝ^{308×256×256} from 8 complex
field measurements {E^l}_{l=0}^{7}, each of shape (256, 256) complex128.

The forward map is the SSNP model:
    E^l = F · P_NA · P_{-Nz/2} · [P · Q]^{Nz} · Φ₀^l(Δn)

This is a nonlinear, strongly ill-posed inverse problem.

## Algorithm

**Gradient descent on complex field MSE with non-negativity projection.**

### Objective

    min_{Δn ≥ 0}  L(Δn) = Σ_{l=0}^{7} ‖E_pred^l(Δn) − E_meas^l‖² / N_pixels

where ‖·‖² = Σ |·|² sums squared moduli of complex residuals.

### Update rule

    Δn_{t+1} = max(0,  Δn_t − η · ∇_{Δn} L)

Gradient ∇_{Δn} L is computed by PyTorch autograd through the complex SSNP operators.

### Hyperparameters

- Iterations: 100
- Learning rate η: 5.0
- TV weight τ: 0.0
- Positivity: True
- Initialisation: Δn = 0

## Pipeline

1. **Preprocessing** — load complex measurements from `raw_data.npz` and
   ground truth from `ground_truth.npz`
2. **Forward model** — build `SSNPForwardModel` from metadata
3. **Reconstruction** — run `SSNPReconstructor` for 100 iterations
4. **Evaluation** — compute NCC and NRMSE vs ground truth
5. **Output** — save `reconstruction.npy`, loss curve, and plots

## Why this suits the problem

- Complex field MSE has a unique global minimum (when measurements exactly determine Δn),
  whereas amplitude-only IDT loss has a phase-ambiguity making it harder to optimise.
- SSNP handles multiple scattering exactly (within the paraxial z-propagation
  approximation), which is critical for this 30.8 μm thick sample.
- Non-negativity is physically motivated: biological cells (n ≈ 1.35–1.40)
  have positive RI contrast relative to the aqueous background (n₀ = 1.0).
- Gradient descent via autograd is the simplest approach; more advanced
  second-order or ADMM methods could improve convergence speed.
