# Approach: MCR Hyperspectral Unmixing

## Problem Statement

Recover 3 pure spectral components and their 50×100 spatial concentration maps from a noisy 5000×200 hyperspectral data matrix, using multivariate curve resolution with alternating regression (MCR-AR).

## Mathematical Formulation

### Forward model

The bilinear mixing model:

D = C S^T + E

- D ∈ ℝ^(5000 × 200): observed noisy data (50×100 spatial pixels, 200 spectral channels)
- C ∈ ℝ^(5000 × 3): concentration matrix
- S^T ∈ ℝ^(3 × 200): spectral matrix (pure spectra as rows)
- E ~ N(0, 250²): i.i.d. Gaussian noise

### Constraints

Physical constraints to resolve rotational ambiguity:
- **Non-negativity**: C ≥ 0, S^T ≥ 0 (concentrations and spectra are physically non-negative)
- **Sum-to-one normalisation**: Σ_k C_{i,k} = 1 for each pixel i (concentrations are fractional)
- **Spectral shape**: optional Gaussian shape enforcement via NLLS fitting

### Alternating regression

At each iteration:
1. Solve C = D (S^T)^† using regressor R_C (OLS, NNLS, or Lasso)
2. Apply constraints to C (non-negativity, normalisation)
3. Solve S^T = C^† D using regressor R_ST (OLS, NNLS, or Ridge)
4. Apply constraints to S^T (non-negativity, optional Gaussian shape)
5. Compute MSE = ||D - C S^T||²_F / (n_pixels × n_freq)

Convergence: iterate until MSE change < 10⁻¹⁴ or max 200 iterations.

## Solution Strategy

### Step 1: Data generation
Generate synthetic HSI with 3 Gaussian spectral components (centres at 1200, 1600, 2000 au) and 2D Gaussian concentration maps with random patches. Add Gaussian noise (σ=250).

### Step 2: Initial spectral guess
Truncated SVD of the noisy data matrix (k+1=4 components). Take absolute values of the top 3 right singular vectors, scaled to the data range.

### Step 3: MCR fitting (5 methods)

| Method | C regressor | S^T regressor | C constraints | S^T constraints |
|--------|------------|---------------|---------------|-----------------|
| MCR-ALS | OLS | OLS | NonNeg + Norm | NonNeg |
| MCR-NNLS | NNLS | NNLS | Norm | (none) |
| MCR-AR Gauss | NNLS | NNLS | NonNeg + Norm | NonNeg + Gauss |
| MCR-AR Ridge | OLS | Ridge(α=10) | NonNeg + Norm | NonNeg |
| MCR-AR Lasso | Lasso(α=1) | OLS | NonNeg + Norm | NonNeg |

### Step 4: Component matching
Match estimated components to true components by minimum squared error assignment (greedy, per-component).

### Step 5: Evaluation
Compute per-method: NCC and NRMSE for concentrations and spectra, MSE convergence, timing.

## Expected Results

| Method | Convergence | Reconstruction MSE |
|--------|------------|-------------------|
| MCR-ALS | Converges early via tol_err_change | ~1.8e5 |
| MCR-NNLS | Runs to max_iter | ~6.3e4 (best) |
| MCR-AR Gauss | Runs to max_iter | ~1.5e6 (highest, but best spectral shapes) |
| MCR-AR Ridge | Converges early | ~1.8e5 |
| MCR-AR Lasso | Slowest per iteration | Variable |

Key insight: MCR-NNLS achieves the lowest reconstruction error; MCR-AR Gauss has highest MSE but recovers the most faithful Gaussian spectral shapes when the model assumption matches reality.
