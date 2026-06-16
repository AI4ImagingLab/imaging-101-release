# Approach: PET MLEM Reconstruction

## Algorithm Overview

We reconstruct a 2D PET activity image from Poisson-noisy sinogram data using two EM-based algorithms that maximize the Poisson log-likelihood.

## MLEM Algorithm

The MLEM update at iteration k:

    x^{k+1}_j = (x^k_j / s_j) * sum_i A_ij * y_i / (sum_j' A_ij' x^k_j' + r_i)

where s_j = sum_i A_ij is the sensitivity (precomputed as A^T 1).

Equivalently in operator form:

    1. Forward project:     p = A x^k + r
    2. Compute ratio:       q = y / p
    3. Back-project ratio:  c = A^T q
    4. Update:              x^{k+1} = x^k * c / s

Implementation uses scikit-image's radon (forward: A x) and iradon with no filter (adjoint: A^T y).

Solver parameters: n_iter = 50, initialized with uniform image x^0 = 1.

## OSEM Algorithm

Divides the n_angles projection angles into n_subsets interleaved subsets for balanced angular coverage. Within each full iteration, performs one MLEM-like update per subset:

    For each subset S:
        1. p_S = A_S x + r_S
        2. q_S = y_S / p_S
        3. c_S = A_S^T q_S
        4. x = x * c_S / s_S

where A_S, y_S, r_S, s_S are restricted to the angles in subset S.

Solver parameters: n_iter = 10, n_subsets = 6.
Effective number of updates = n_iter * n_subsets = 60.

## Evaluation

Metrics computed within the activity mask (ground truth > 0):
- NCC (cosine similarity) between estimated and ground truth activity maps
- NRMSE normalized by dynamic range of ground truth
- Boundary: 90% of best NCC, 110% of best NRMSE
