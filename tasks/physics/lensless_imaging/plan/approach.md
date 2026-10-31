# Approach: Lensless Imaging Reconstruction via ADMM

## Problem

Recover an RGB image `v ∈ ℝ^{H×W×3}` from a lensless (DiffuserCam) sensor
measurement `b ∈ ℝ^{H×W×3}` given the calibration PSF `h ∈ ℝ^{H×W×3}`.

The forward model is:

    b = A v + n,      A v ≜ crop( h * v )

where `*` is 2-D linear convolution and `crop` trims the result to the sensor
size `(H, W)`.  The noise `n` is assumed to be small (the system is
noise-limited rather than ill-posed per se, but the crop causes rank deficiency).

We solve the regularised problem:

    min_{v ≥ 0}  (1/2) ||b - Av||²₂  +  τ ||Ψv||₁          (★)

where `Ψv = [∂ₓv; ∂ᵧv]` is the image gradient (finite differences) and
`||·||₁` is the isotropic total variation (TV) penalty that promotes
piecewise-smooth reconstructions.

## Variable Splitting and Augmented Lagrangian

We decompose `A = CM` where `M` is the full (uncropped) convolution
and `C` is the crop operator.  Introduce auxiliary variables:

    x = Mv   (forward-model variable, full padded size)
    u = Ψv   (gradient variable, enforces TV sparsity)
    w = v    (non-negativity variable)

Problem (★) becomes:

    min_{v,u,x,w}  (1/2)||b - Cx||²₂ + τ||u||₁ + 𝟙₊(w)
    s.t.  x = Mv,  u = Ψv,  w = v

Augmented Lagrangian (with dual variables ξ, η, ρ):

    L = (1/2)||b - Cx||² + τ||u||₁
        + (μ₁/2)||Mv - x||² + ξᵀ(Mv - x)
        + (μ₂/2)||Ψv - u||² + ηᵀ(Ψv - u)
        + (μ₃/2)||v - w||²  + ρᵀ(v - w)
        + 𝟙₊(w)

## ADMM Updates

Each iteration k performs:

### 1. u-update (TV soft-thresholding)

    u_{k+1} = S_{τ/μ₂}( Ψv_k + η_k/μ₂ )

where  S_λ(x) = sign(x) · max(|x| - λ, 0)  is element-wise soft-thresholding.
This is the proximal operator of (τ/μ₂)||·||₁.

### 2. x-update (data consistency, element-wise division)

    x_{k+1} = (CᵀC + μ₁I)⁻¹ (ξ_k + μ₁Mv_k + Cᵀb)

`CᵀC` is a diagonal matrix: 1 in the sensor (crop) region, 0 outside.
Hence `(CᵀC + μ₁I)⁻¹` is also diagonal:

    x_i = (ξ_k,i + μ₁(Mv_k)_i + b_i) / (1 + μ₁)    if i ∈ crop
    x_i = (ξ_k,i + μ₁(Mv_k)_i)       / μ₁            otherwise

### 3. w-update (non-negativity projection)

    w_{k+1} = max( v_k + ρ_k/μ₃ , 0 )

This is the proximal operator of 𝟙₊(·).

### 4. v-update (frequency-domain solve)

    v_{k+1} = F⁻¹{ R_divmat · F{r_k} }

where

    r_k = (μ₃w_{k+1} - ρ_k) + Ψᵀ(μ₂u_{k+1} - η_k) + Mᵀ(μ₁x_{k+1} - ξ_k)

and the precomputed denominator is (element-wise in frequency domain):

    R_divmat = 1 / ( μ₁|H̃|² + μ₂|Ψ̃|² + μ₃ )

Here `|H̃|²` is the magnitude-squared of the FFT of h (padded), and
`|Ψ̃|²` is the eigenvalue spectrum of ΨᵀΨ (the discrete Laplacian):

    |Ψ̃(k₁,k₂)|² = 4 - 2cos(2πk₁/N₁) - 2cos(2πk₂/N₂)

computed via `rfft2` of the Laplacian kernel `[4, -1, 0, …, -1; -1, …]`.

### 5. Dual updates

    ξ_{k+1} = ξ_k + μ₁(Mv_{k+1} - x_{k+1})
    η_{k+1} = η_k + μ₂(Ψv_{k+1} - u_{k+1})
    ρ_{k+1} = ρ_k + μ₃(v_{k+1} - w_{k+1})

## Efficient Implementation

- All convolutions are done via `rfft2` / `irfft2` (real-valued FFT) on
  zero-padded arrays of size `next_fast_len(2H-1) × next_fast_len(2W-1)`.
- `R_divmat`, `X_divmat`, `H_fft`, and `Cᵀb` are all precomputed in `reset()`.
- Each ADMM iteration costs O(HW log HW) operations (dominated by FFTs).

## Initialisation and Hyperparameters

| Parameter | Default value | Role                        |
|-----------|---------------|-----------------------------|
| μ₁        | 1e-6          | Penalty for x = Mv          |
| μ₂        | 1e-5          | Penalty for u = Ψv (TV)     |
| μ₃        | 4e-5          | Penalty for w = v (nonneg)  |
| τ         | 2e-5          | TV regularisation weight    |
| n_iter    | 100           | Number of ADMM iterations   |

Initial estimate: `v₀ = 0.5 · 1` (half-intensity image).
All dual variables: initialised to zero.

## Expected Quality

After 100 iterations the reconstruction is visually interpretable and
the reprojection residual `||Av̂ - b|| / ||b||` is typically ≲ 0.5.
More iterations (300–500) yield sharper results; the LenslessPiCam
paper reports PSNR ≈ 11.7 dB on the DiffuserCam Mirflickr dataset.
