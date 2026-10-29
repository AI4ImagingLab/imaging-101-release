# Approach: Conventional Ptychography Reconstruction via mPIE

## Problem Statement

Conventional ptychography (CP) is a computational imaging technique that reconstructs
a complex-valued sample transmission function O(r) and probe wavefield P(r) from a
series of far-field diffraction patterns. The sample is raster-scanned, and at each
scan position j the exit surface wave is:

    ψ_j(r) = P(r) · O(r − r_j)

The measured intensity at the detector is:

    I_j(q) = |D_{r→q}[ψ_j](q)|²

where D is the propagation operator (Angular Spectrum or Fraunhofer).

Since only the intensity is measured (not the phase), this is a phase retrieval problem.
The redundancy from overlapping scan positions enables simultaneous recovery of both
O and P via iterative algorithms.

## Dataset

Synthetic simulation using standard CP parameters:
- **Wavelength**: λ = 632.8 nm (He-Ne laser)
- **Sample–detector distance**: z_o = 50 mm
- **Detector**: 128 × 128 pixels, dxd = 72 μm effective pixel size
- **Probe pixel size**: dxp = λ z_o / (Nd · dxd) ≈ 3.433 μm
- **Object**: 542 × 542 pixels (No = 542), dxo = dxp
- **Probe**: Circular aperture with quadratic focusing phase
- **Scan grid**: Non-uniform Fermat spiral (100 positions, radius 150 px)
- **Noise**: Poisson noise (14-bit dynamic range)

## Algorithm: Momentum-accelerated PIE (mPIE)

mPIE [Maiden et al., Optica 4, 736 (2017)] is the standard CP solver in PtyLab.
It combines the ePIE update rule with Nesterov-style global momentum acceleration.

### ePIE update (per scan position j)

For each position, form the exit wave:
    ψ_j = P · O_j

Apply the intensity constraint to get the updated detector wave (standard):
    ψ̃_j(q) = sqrt(I_j(q)) · Ψ_j(q) / |Ψ_j(q)|

Back-propagate to get the updated exit wave ψ̂_j.

Update the object and probe via ePIE:
    O_{n+1} = O_n + β · P_n* · (ψ̂_j − ψ_j) / max(|P_n|²)
    P_{n+1} = P_n + β · O_j* · (ψ̂_j − ψ_j) / max(|O_j|²)

The ePIE regularization (Thibault & Menzel, Nature 494, 2013) uses:
    Γ_ePIE = diag[(max(|P|²)/αβ − |P|²/α)^(1/2)]

### Momentum step (after T ePIE iterations)

After every T = 50 ePIE iterations, apply a momentum update to the full object:
    v_n = η · v_{n-T} + O_{n,oFOV} − O_{n+1-T,oFOV}    [η = 0.7]
    O_{n+1,oFOV} = O_{n,oFOV} + η · v_n

The momentum term accelerates convergence by exploiting gradient history.

### Autofocus and regularization schedule

Following the standard mPIE schedule:
- Odd rounds: no L2 regularization
- Even rounds: L2 probe and object regularization (aleph = 1e-2)

Total: 7 rounds × 50 iterations = 350 iterations.

## Convergence Metric

The reconstruction error is the normalized amplitude discrepancy:

    E = Σ_j Σ_q (sqrt(I_meas,j) − sqrt(I_est,j))² / Σ_j Σ_q I_meas,j

Target: E < 0.05 after 350 iterations on the simulated dataset.

## Key References

- Rodenburg & Faulkner, Phys. Rev. Lett. 93, 023903 (2004) — PIE
- Maiden & Rodenburg, Ultramicroscopy 109, 1256 (2009) — ePIE
- Maiden, Johnson & Li, Optica 4, 736 (2017) — mPIE
- Loetgering et al., Opt. Express 31, 13763 (2023) — PtyLab toolbox
