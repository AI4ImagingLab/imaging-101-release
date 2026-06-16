# MCR Hyperspectral Unmixing

> Recover pure spectral components and their spatial concentration maps from a noisy hyperspectral image using multivariate curve resolution, where the bilinear mixing model and rotational ambiguity make the factorisation ill-posed.

> Domain: Chemistry | Keywords: hyperspectral imaging, spectral unmixing | Difficulty: Medium

## Background

Hyperspectral imaging (HSI) captures a full spectrum at every spatial pixel, producing a 3D data cube. In many chemical imaging applications (Raman, IR, fluorescence), the sample consists of a small number of pure chemical components whose spectra overlap spatially. The goal of **multivariate curve resolution (MCR)** is to decompose the observed data into the underlying pure spectra and their spatial concentration maps — a blind source separation problem.

MCR assumes a **bilinear mixing model**: each pixel's measured spectrum is a weighted sum of the pure component spectra. The weights are the fractional concentrations of each component at that pixel. Because the factorisation is not unique (rotational ambiguity), physical constraints — non-negativity of spectra and concentrations, sum-to-one normalisation of concentrations, spectral shape priors — are essential for obtaining chemically meaningful solutions.

**MCR-AR (alternating regression)** generalises MCR-ALS by allowing arbitrary regressors (Ridge, Lasso, NNLS) and constraints at each alternating step. This task benchmarks five MCR-AR variants from the pyMCR paper (Camp 2019, J. Res. NIST) on a synthetic 3-component system.

## Problem Description

The forward model is:

$$\mathbf{D} = \mathbf{C} \mathbf{S}^T + \mathbf{E}$$

where:
- **D** ∈ ℝ^(n_pixels × n_freq) is the observed hyperspectral data matrix (pixels unraveled from 2D)
- **C** ∈ ℝ^(n_pixels × k) is the concentration matrix (k components)
- **S^T** ∈ ℝ^(k × n_freq) is the spectral matrix (pure spectra as rows)
- **E** is additive Gaussian noise

The inverse problem is: given noisy **D**, recover **C** and **S^T** simultaneously, subject to physical constraints. The problem is ill-posed because any invertible matrix **T** satisfies **D = (CT)(T⁻¹S^T)**, so constraints are needed to resolve the rotational ambiguity.

MCR-AR solves this by alternating regression:
1. Fix **S^T**, solve for **C** (regress columns of D onto rows of S^T)
2. Apply constraints to **C** (non-negativity, normalisation)
3. Fix **C**, solve for **S^T** (regress rows of D onto columns of C)
4. Apply constraints to **S^T** (non-negativity, shape priors)
5. Repeat until convergence

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| hsi_noisy | (1, 5000, 200) | float64 | Noisy hyperspectral data, 50×100 pixels unraveled, 200 spectral channels (intensity, au) |
| wn | (1, 200) | float64 | Wavenumber axis from 400 to 2800 (au) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| concentrations | (1, 50, 100, 3) | float64 | True 2D concentration maps, sum-to-one at each pixel |
| concentrations_ravel | (1, 5000, 3) | float64 | True concentrations unraveled to 2D |
| spectra | (1, 3, 200) | float64 | True pure spectral components (intensity, au) |
| hsi_clean | (1, 5000, 200) | float64 | Noise-free hyperspectral data (intensity, au) |

### data/meta_data.json

Imaging parameters: spatial dimensions (M=50, N=100), number of components (3), spectral range, noise level (σ=250), and spectral peak parameters.

## Method Hints

- Use **truncated SVD** of the noisy data matrix to obtain an initial spectral guess (take top k+1 singular vectors, drop the smallest, scale by data maximum).
- Compare five MCR-AR configurations that differ in regressor (OLS, NNLS, Ridge, Lasso) and constraints (non-negativity, normalisation, Gaussian shape enforcement on spectra).
- The Gaussian shape constraint fits each spectral row to a Gaussian + constant model via nonlinear least-squares (lmfit), which acts as a strong regulariser.
- Ridge regression on S^T penalises large spectral values; Lasso regression on C promotes sparse concentrations.
- MCR-NNLS typically achieves the lowest reconstruction MSE; MCR-AR Gauss recovers the most accurate spectral shapes when the true spectra are indeed Gaussian.

## References

- Camp, C.H. (2019). "pyMCR: A Python Library for Multivariate Curve Resolution and Related Constrained Factorizations." *Journal of Research of NIST*, 124, 124018.
- pyMCR GitHub: https://github.com/usnistgov/pyMCR
