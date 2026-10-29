# Approach: EHT Black Hole Tomography (BH-NeRF)

## Problem Statement
Recover the 3D volumetric emission field e(x,y,z) of a black hole accretion flow from time-varying 2D images I(t), using the physical constraint that the emission follows Keplerian orbital dynamics.

## Key Insight
The emission field is approximately static in a co-rotating frame: inner orbits rotate faster than outer orbits (Keplerian shearing). A single static 3D emission field, combined with known orbital dynamics, fully determines the time-varying movie. This converts the ill-posed 2D→3D problem into a well-constrained optimization.

## Solution Strategy

### 1. Schwarzschild Ray Tracing (Preprocessing)
Pre-compute gravitationally lensed ray paths through the Schwarzschild metric for each image pixel. These curved paths encode the 3D-to-2D projection including gravitational lensing effects. Also pre-compute Doppler boosting factors and integration weights along each ray.

### 2. NeRF Emission Representation
Represent the static 3D emission field as a coordinate-based MLP neural network with positional encoding (L=3). The network maps 3D coordinates to emission density: e_0(x) = sigmoid(MLP(gamma(x)) - 10).

### 3. Keplerian Velocity Warp
For each time frame t, apply an inverse Keplerian rotation to the ray coordinates before querying the NeRF. The rotation angle depends on radius: theta(r,t) = Omega(r) * t where Omega(r) = r^{-3/2}. This produces the time-varying emission field from the static NeRF.

### 4. Volume Rendering
Integrate the time-warped emission along each gravitationally lensed ray path using the radiative transfer equation: I_n(t) = sum(g^2 * e(t, x_i) * dtau_i * Sigma_i).

### 5. Optimization
Minimize the L2 loss between predicted and observed images using Adam with a polynomial learning rate schedule. Jointly optimize the MLP weights (theta) and the rotation axis (xi). Use multi-initialization from both hemispheres to avoid symmetric local minima.

## Expected Results
- NRMSE < 0.5 on 3D emission recovery
- NRMSE < 0.3 on image-plane movie
- Training converges in ~5000 iterations
