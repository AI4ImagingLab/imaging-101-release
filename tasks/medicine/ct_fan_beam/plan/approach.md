# Approach: Fan-Beam CT Reconstruction

## Algorithm Overview

We reconstruct a 2D attenuation image from fan-beam CT sinograms using two methods:
1. Fan-beam filtered back-projection (FBP) with Parker weighting for short-scan
2. TV-regularized iterative reconstruction via Chambolle-Pock (PDHG)

## Fan-Beam FBP

The fan-beam FBP formula with flat detector geometry:

    f(x, y) = integral_0^{2pi} (1/U^2) * Q(beta, t_proj) d_beta

where:
- U = D_sd - s (distance from source to pixel along source-detector axis)
- s = -x*sin(beta) + y*cos(beta) (perpendicular coordinate)
- t_proj = t * (D_sd + D_dd) / U (magnified detector coordinate)
- Q = filtered, pre-weighted sinogram

### Step 1: Parker Weighting (short-scan only)

For angular range delta = pi + 2*gamma_max, Parker weights are:

    w(beta, gamma) = sin^2(pi/2 * beta/(2*(epsilon+gamma)))       if beta < 2*(epsilon+gamma)
    w(beta, gamma) = sin^2(pi/2 * (delta-beta)/(2*(epsilon-gamma))) if beta > delta-2*(epsilon-gamma)
    w(beta, gamma) = 1                                              otherwise

where epsilon = gamma_max and gamma = arctan(t/D_sd).

### Step 2: Pre-weighting

Each sinogram row is multiplied by the distance-dependent weight:

    w(gamma) = D_sd / sqrt(D_sd^2 + gamma_iso^2)

where gamma_iso = t * D_sd / (D_sd + D_dd) is the detector position projected to the isocenter plane.

### Step 3: Ramp Filtering

The ramp filter is constructed in the spatial domain:

    h[0] = 1/(8*d^2)
    h[n] = -1/(2*pi*d*n)^2  for odd n
    h[n] = 0                 for even n != 0

Applied in frequency domain via FFT, with optional windowing (Hann: cutoff=0.3).

### Step 4: Distance-Weighted Back-Projection

For each angle beta and each pixel (x,y):
- Compute rotated coordinates: t = x*cos(beta) + y*sin(beta), s = -x*sin(beta) + y*cos(beta)
- Compute magnified detector coordinate: t_proj = t*(D_sd+D_dd) / (D_sd-s)
- Compute distance weight: (D_sd)^2 / (D_sd-s)^2
- Interpolate filtered sinogram at t_proj and accumulate

## TV-PDHG Iterative Reconstruction

Minimizes:  (1/2) ||A x - b||^2 + lam * TV(x)

Using Chambolle-Pock primal-dual splitting:
- Dual update for data fidelity: q <- (q + sigma*(A*x_bar - b)) / (1 + sigma)
- Dual update for TV: p <- prox_{sigma*lam*||.||_1}(p + sigma*grad(x_bar))
- Primal update: x <- x - tau*(A^T*q - div(p))
- Over-relaxation: x_bar <- x + theta*(x - x_old)

Initialized with FBP. Positivity constraint enforced.

Solver parameters: lam=0.005, n_iter=150, tau=0.005, sigma=0.005, theta=1.0

## Evaluation

Metrics computed on centre-cropped (80%), normalized reconstructions:
- NCC (cosine similarity) and NRMSE vs ground truth
- Boundary: 90% of best NCC, 110% of best NRMSE
