# Approach: MRI CG-SENSE Reconstruction

## Problem Statement
Reconstruct a 128x128 brain MRI from 4x-accelerated 8-coil k-space using CG-SENSE.

## Algorithm: Conjugate Gradient SENSE

### Encoding model
$$y = Ax, \quad A_c(x) = M \cdot \mathcal{F}(S_c \cdot x)$$

### Normal equations
$$A^H A \hat{x} = A^H y$$

where $A^H y = \sum_c \bar{S}_c \mathcal{F}^{-1}(y_c)$ and $A^H A x = \sum_c \bar{S}_c \mathcal{F}^{-1}(M \cdot \mathcal{F}(S_c x))$.

### Solver
Conjugate Gradient iteration (ported from scipy.sparse.linalg.cg) with $E = A^H A$ applied as a matrix-free matvec callback. Handles complex Hermitian systems via conjugate inner products ($\text{vdot}$). No regularization (pure CG-SENSE). Default tolerance: $\|r\| < 10^{-5} \|b\|$.

## Expected Results
- CG-SENSE: SSIM=0.719, NCC=0.988, NRMSE=0.057
- Zero-fill: SSIM=0.541, NCC=0.900, NRMSE=0.110
