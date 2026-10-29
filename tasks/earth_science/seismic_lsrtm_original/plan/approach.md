# LSRTM Approach (From-Scratch C-PML)

## Inverse Problem
Recover the scattering image `s(y, x)` from observed shot gathers by subtracting the direct arrivals predicted by the smooth migration velocity `v_mig(y, x)` and fitting the residual data with a linearized acoustic Born model.

## Pipeline
1. Load `v_mig` and `observed_data` from `data/raw_data.npz` and load acquisition parameters from `data/meta_data.json`.
2. Build source and receiver geometry from the metadata.
3. Forward-model the smooth-background response and compute scattered data `d_scat = d_obs - d_0`.
4. Optimize the scattering image with L-BFGS to minimize `0.5 * ||L(v_mig)s - d_scat||_2^2`.
5. Save reconstructed images, predicted scattered data, loss history, and diagnostic figures.

## Numerical Model
- Background propagation uses a from-scratch acoustic finite-difference solver with C-PML absorbing boundaries.
- Born modeling evolves both background and scattered wavefields and injects the linearized coupling term at every time step.
- Receivers sample the scattered wavefield only, matching the LSRTM objective.

## Solver Constants
The task keeps solver settings out of `meta_data.json`. The cleaned entry point defines them as named constants or command-line defaults:

- `_DEFAULT_EPOCHS = 3`
- Optimizer: `L-BFGS`
- Loss: data least squares on scattered gathers

## Parity Expectations
The cleaned C-PML solver is intended to remain numerically consistent with the original research implementation and with the Deepwave-based reference setup at the level of direct modeling, Born modeling, and LSRTM image formation.
