# Oysand Rayleigh Shear Profile

> Reconstruct a layered shear-velocity field beneath a real surface-wave survey from its complete fundamental-mode Rayleigh dispersion curve, with depth and velocity tradeoffs.
> Domain: earth science | Keywords: seismic imaging, surface waves, elastic inversion | Difficulty: Hard

## Background

Rayleigh waves propagate along an elastic surface while their displacement decays with depth. Their speed depends on compressional and shear elasticity and density. Different wavelengths sample different depth ranges, so a measured frequency-dependent phase velocity provides indirect information about subsurface structure.

This modality supports near-surface geotechnical characterization and studies of sedimentary layering. A finite dispersion curve has limited depth resolution: different velocity profiles and interface depths can produce similar observations. A parameterization and independently stated elastic assumptions are therefore part of the physical interpretation.

## Problem Description

Use all30 real Oysand composite fundamental-mode Rayleigh observations to infer a horizontally stratified four-layer shear-velocity field. Three finite layers lie over a half-space. Unknowns are their positive thicknesses h1,h2,h3 and the four positive shear velocities Vs1,...,Vs4. Density and Poisson ratio are fixed known assumptions. With depth z positive downward, Vs(z) is piecewise constant, with interfaces at h1,h1+h2,h1+h2+h3. The half-space extends indefinitely.

Use the conditional structural prior Vs1<=Vs2<=Vs3<=Vs4. Admissible thickness intervals are(0.4,1.2),(0.5,2.0),(4,12)m; corresponding velocity intervals are(80,160),(100,180),(120,220),(150,280)m/s. These define the allowed layered parameterization, not measurements of the true geology.

For period T, let angular frequency omega=2*pi/T and horizontal wavenumber k=omega/c, where c is the predicted phase velocity. Compressional speed is Vp=Vs*sqrt((1-nu)/(0.5-nu)), where nu is Poisson ratio. The elastic displacement u obeys

```math
-\rho\omega^2\mathbf u=(\lambda+\mu)\nabla(\nabla\cdot\mathbf u)+\mu\nabla^2\mathbf u,
\qquad \mu=\rho V_s^2,\quad \lambda=\rho(V_p^2-2V_s^2).
```

Here rho is density and lambda,mu are the Lame moduli. The surface has zero normal and shear traction. Displacement and traction are continuous at interfaces; the bottom half-space admits decaying waves. These conditions define a secular function D(c,T;h,Vs), whose fundamental-mode zero gives c. The observations satisfy measured c(T)=predicted c(T)+error, with supplied uncertainty widths defining residual weights. Recovering interface depths and velocities jointly is nonlinear and nonunique over a finite period band.

Save `output/reconstruction.npz` relative to the task workspace, containing both required real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,4) | float64 | Shear velocities in m/s, ordered surface layer to bottom half-space; strictly positive. |
| thickness | (1,3) | float64 | Finite-layer thicknesses in m, ordered surface downward; strictly positive. |

These seven independent physical quantities define the full depth field; no arbitrary depth sampling grid is required. Evaluation checks the absolute error of every velocity in m/s and every thickness in m, plus the RMS of all30 calculated-minus-measured phase velocities divided by their supplied uncertainty widths. There is no layer permutation, depth shift, rescaling or curve alignment. Reference agreement measures a validated conditional method result, not a measured geological truth.

## Data Description

`data/raw_data.npz` contains the complete real composite dispersion curve, with batch axis first:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| period | (1,30) | float64 | Wave periods in s, increasing order; equal to original wavelength divided by phase velocity. |
| phase_velocity | (1,30) | float64 | Observed fundamental Rayleigh phase velocity in km/s, matching period order. |
| uncertainty | (1,30) | float64 | Positive phase-velocity weight scale in km/s, half the published upper-minus-lower uncertainty interval. |

Acquisition metadata:

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Site | Oysand, Norway | - | Real near-surface surface-wave survey; released composite dispersion observations. |
| Mode | Fundamental Rayleigh phase | - | A single complete selected mode; not group speed or a higher mode. |
| Coordinate | z positive downward from surface | m | Horizontal layers; free surface at z=0. |
| Layers | 3finite plus1half-space | - | Four independently specified shear speeds, three finite interface increments. |
| Poisson ratio | 0.30,0.30,0.49,0.49 | dimensionless | Fixed by layer, surface downward. |
| Density | 1.95 in every layer | g/cm^3 | Fixed elastic prior, equal to1950kg/m^3. |
| Compressional speed | Vs*sqrt((1-nu)/(0.5-nu)) | same speed unit as Vs | Fixed isotropic-elastic relation. |
| Surface boundary | Zero normal/shear traction | - | Stress-free elastic surface. |
| Interface boundary | Continuous displacement/traction | - | Perfectly bonded interfaces. |
| Bottom boundary | Decaying elastic wave | - | Infinite half-space; no artificial reflecting bottom. |
| Uncertainty interpretation | Composite interval half-width | km/s | Supplied weighting scale, not a claim of independent Gaussian standard errors. |

The observations are supplied as processed dispersion measurements, not raw seismograms. Wave wavelengths are recoverable as period times phase speed after consistent unit conversion. No geological truth, target-fitted calibration or hidden spatial profile is provided. Density and Poisson ratios are modeling assumptions, so a recovered profile is conditional on them.

## Method Hints

Use a stable layered elastic-wave secular function and bracket the fundamental phase-velocity root for each period. A compound-matrix representation avoids unstable exponential cancellation in thick or high-frequency layers. Joint thickness/velocity search is multimodal; a bounded particle-swarm family with monotonic-velocity prior and competitive restarts is suitable. Minimize uncertainty-weighted dispersion RMS while keeping all periods and all seven physical unknowns explicit. Numerical restart radius conventions depend on optimization coordinates and must be declared.

## References

- Luu et al.(2018), *A parallel competitive Particle Swarm Optimization for non-linear first arrival traveltime tomography and uncertainty quantification*, Computers & Geosciences113,81-93. [Article](https://doi.org/10.1016/j.cageo.2018.01.016). Author's full article chapter in [2018 thesis](https://pastel.hal.science/tel-02173705/document).
- Dunkin(1965), *Computation of modal solutions in layered, elastic media at high frequencies*, Bulletin of the Seismological Society of America55,335-358. [Article](https://doi.org/10.1785/BSSA0550020335).
- Olafsdottir et al.(2024), *A Tool for Processing and Inversion of MASW Data and a Study of Inter-Session Variability of MASW*, Geotechnical Testing Journal47(5),1006-1025; author-released composite Oysand data. [Article](https://doi.org/10.1520/GTJ20230380).
