# Implemented mathematics and numerical conventions

The selected implementation is the PWL option of the published ridge-regularized DRT method, not the Gaussian-RBF option. There are N=107 frequency samples f_i, tau_j=1/f_j and u_j=ln(tau_j). Quadrature weights are w_0=(u_1-u_0)/2, w_(N-1)=(u_(N-1)-u_(N-2))/2 and w_j=(u_(j+1)-u_(j-1))/2 otherwise. A_ij=w_j/(1+i 2π f_i tau_j). The complete design B has columns [i2πf,1,A]; x=[L,R,gamma]. All unknowns are constrained nonnegative.

The exact objective is ||Re(Bx-z)||²+||Im(Bx-z)||²+lambda||Dx||², with no division by sample count and no factor one-half. lambda=1e-3. D ignores L,R and has adjacent entries [-1/delta_j,+1/delta_j], delta_j=ln(tau_j/tau_(j+1)); the author's opposite sign convention to increasing log-time cancels in DᵀD and is preserved. Solve the explicitly augmented real system [Re B;Im B;sqrt(lambda)D]x≈[Re z;Im z;0] using general-purpose scipy.optimize.nnls, maxiter=10000. NNLS uses its active-set initialization/stopping; there is no custom line search or arbitrary iteration loop.

The independent oracle solves the identical nonnegative quadratic program with CVXOPT. Its first numerical stopping settings left a small positive barrier in near-zero gamma coefficients; tightening oracle absolute tolerance to 1e-18, relative tolerance to 1e-15, feasibility tolerance to 1e-14, maxiters=200 reconciled the two optimizers without changing lambda, model or acceptance tolerance. The unconstrained posterior mean stored internally by upstream is not its published nonnegative output and is not used as reference.

Evaluate complex residuals in ohm. Plot Nyquist minus-imaginary values only for display; submitted arrays retain the original imaginary sign. No normalization, alignment or log conversion is applied by Judge.

## Evaluation

No independent object truth exists for these bound observations. The hidden baseline is the freshly executed original selected method. Judge uses strict per-array/per-component relative Euclidean error with boundary 1e-5 fixed before extraction; complex vectors use their Hermitian norm. Original and fresh measurements are identified separately. Do not interpret reference agreement as physical accuracy. Stage, analytic physics and Judge contract tests complement full-pipeline parity.
