# Approach: Deep Probabilistic Imaging (DPI)

## Problem Statement

Given sparse, noisy interferometric closure measurements from the EHT,
learn the full posterior distribution p(image | data) rather than a single
point estimate. This enables principled uncertainty quantification.

## Algorithm

### 1. Posterior Approximation via Normalizing Flow

DPI uses a Real-NVP normalizing flow G_θ that maps a standard Gaussian
latent space to the image posterior:

    z ~ N(0, I)  →  x = G_θ(z)  ~  q_θ(x) ≈ p(x | y)

The flow is trained by minimizing the KL divergence KL(q_θ || p(·|y)),
which reduces to (Eq. 7 of Sun & Bouman 2020):

    θ* = argmin E_z [ L_data(y, f(G_θ(z))) + λ R(G_θ(z)) - β log|det(∂G_θ/∂z)| ]

### 2. Data Fidelity (Closure Quantities)

Closure phases and log closure amplitudes are used instead of raw
visibilities because they are immune to station-based gain errors:

- Closure phase loss: 2 * mean((1 - cos(φ_true - φ_pred)) / σ²)
- Log closure amp loss: mean((lnA_true - lnA_pred)² / σ²)

### 3. Image Priors

- Maximum Entropy (MEM): cross-entropy with Gaussian prior image
- Total Squared Variation (TSV): smoothness
- L1 sparsity: compactness
- Flux constraint: total flux conservation
- Centering: center-of-mass penalty

### 4. Entropy Term

The log-determinant term -β log|det(∂G_θ/∂z)| is critical: it encourages
the flow to maintain diversity in generated samples. Without it, the model
collapses to a point estimate (delta function posterior).

### 5. Positivity via Softplus

Images must be non-negative. DPI applies Softplus(x) * scale_factor to
the flow output, with a log-determinant correction for the transform:

    det_softplus = sum(x - Softplus(x))  [= -sum(log(sigmoid(x)))]

### 6. Posterior Sampling

After training, generate posterior samples by:
1. Sample z ~ N(0, I)
2. Push through flow: x = G_θ(z)
3. Apply Softplus and scale

No MCMC or iterative optimization needed — instant samples.

## Expected Results

- Posterior mean should recover the ring/crescent structure (NRMSE < 0.5)
- Posterior std should be higher in poorly-constrained regions
- Individual samples should show plausible variations
- Calibration: ~68% of GT pixels within 1-sigma of posterior mean
