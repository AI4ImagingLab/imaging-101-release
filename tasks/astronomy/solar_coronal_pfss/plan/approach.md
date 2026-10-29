# Implemented finite-volume coronal field

The unknown is the complete signed magnetic vector in a spherical shell, conditional on current-free coronal physics and a radial source surface. It is not a true observed corona. The input is the full native GONG radial synoptic map,180 sine-latitude by360 longitude samples, in Gauss, stored big-endian float32 exactly as in the FITS image. No pixel crop, resampling, synthetic object or downcast occurs. Computation promotes to native float64; complex spectral work is complex128. One Gauss is1e−4tesla.

## Coordinates and boundary preparation

Use radius in solar-radius units. Set inner radius1, source radiusR=2.5, radial cellsK=35. Let J=180 latitude cells and I=360 longitude cells. Define Δs=2/J,Δφ=2π/I,Δρ=log(R)/K. Radial faces have ρ(k)=kΔρ and r(k)=exp(ρ(k)); radial centers have ρ=(k+1/2)Δρ. Latitude faces s(j)=−1+jΔs and centers s(c,j)=−1+(j+1/2)Δs; s=cosθ is sine(latitude) and increases south to north. θ is colatitude, so positive Bθ points south. Longitude faces are22°+iΔφ and centers22°+(i+1/2)Δφ modulo360°. Phi increases with Carrington longitude, with periodic seam. Original headerCDELT2=.0111111 is retained verbatim in README, while the model's exact complete-sphere increment is2/180; the distinction is explicit and tested through the original run.

Promote the measured map tofloat64 and subtract its float64 arithmetic mean, equal to its surface-area mean on this grid. This flux balancing uses observations only and removes the unphysical magnetic monopole. The measured map is not silently overwritten. The original algorithm also excludes the constantm=0latitude eigenmode. Actual mean and boundary residual measurements are recorded in validation; no hidden target defines calibration.

## Continuum model and discrete eigenproblem

In the shell, div B=0,curl B=0,so B=−gradΦ with LaplaceΦ=0. Inner Br is measured balanced flux density; outer Bθ=Bφ=0, longitude periodic and poles regular. The code uses an auxiliary discrete potentialψ to construct edge-integrated vector potential; ψ is not mislabeled as the continuum scalarΦ.

Let s(g,j),s(c,j) denote faces and centers. Define latitude edge factors

```math
F_p(j)=\frac{\sqrt{1-s(g,j)^2}\,\Delta\phi}
 {\arcsin s(c,j)-\arcsin s(c,j-1)},\quad 1\leq j<J,
```

with Fp=0 at both poles, and

```math
F_s(j)=\frac{\arcsin s(g,j+1)-\arcsin s(g,j)}
 {\sqrt{1-s(c,j)^2}\,\Delta\phi},\quad 0\leq j<J.
```

Set V(j)=Fp(j)/(ΔsΔφ),U(j)=Fs(j)/(ΔsΔφ). Fourier-transform the balanced photospheric Br along longitude with the standard negative-exponential DFT. For mode m=0..I/2, periodic finite-difference longitude eigenvalue is μ(m)=4sin²(πm/I). Form a symmetric real tridiagonal latitude matrix A with diagonal V(j)+V(j+1)+U(j)μ(m) and off-diagonals−V(j+1). Compute its ordered eigenvaluesλ(l) and orthonormal columnsq(l) using numpy.linalg.eigh. No iterative inverse or imaging-domain library is called.

Set e=exp(Δρ),C=sinh(Δρ)(e−1),F(l)=[1+e+λ(l)C]/2. Radial recurrence roots are z+(l)=F+sqrt(F²−e), z−(l)=e/z+. Let b(m) be the longitude-FFT boundary column. Its coefficient is a(l,m)=q(l)ᵀb(m)/λ(l). Omit l=0,m=0, where λ is the zero-monopole singularity. For every other mode define

```math
t=\frac{z_-^{K-1}-z_-^K}{z_+^K-z_+^{K-1}},\qquad
d=\frac{a}{1+t},\qquad c=td,\qquad
\psi(k,j,m)=\sum_l\left(cz_+^k+dz_-^k\right)q_l(j).
```

This ratio is the original numerically safer orientation and imposes the source-surface radial condition using the staggered recurrenceψ(K)=ψ(K−1). Populate negative modes by conjugacy, then inverse-transform longitude with1/I normalization and take the real part. The Nyquist mode follows the same original convention. No line search, initial estimate, stopping rule or regularization weight is needed: this is a deterministic linear boundary solve. Source radius is a physical model assumption; it is not optimized against hidden answers.

## Edge potential, fluxes and signed field components

Let As and Ap denote vector-potential line integrals on s- and φ-directed edges; radial A iszero. For longitude edge i,

As(i,j,k)=Fs(j)[ψ(k,j,i−1)−ψ(k,j,i)] with periodic indexing.
For latitude face j=1..J−1,
Ap(i,j,k)=Fp(j)[ψ(k,j,i)−ψ(k,j−1,i)], with zero polar Ap.

Stokes' theorem gives radial face flux fr=As(i+1)−As(i)+Ap(j)−Ap(j+1), northward face flux fs=Ap(k+1)−Ap(k), and longitudinal face flux fp=As(k)−As(k+1). Divide by the following positive cell-face areas, in solar-radius squared:

- Radial area Ar=r(face)²ΔsΔφ.
- Constantθ area At=.5[r(upper)²−r(lower)²]Δφsqrt(1−s(face)²).
- Constantφ area Aparea=.5[r(upper)²−r(lower)²][arcsin(s(upper))−arcsin(s(lower))].

At the coordinate-singular polar faces, use the adjacent latitude face's area and the original regularity convention fs(pole,i)=.5[fs(adjacent,i)−fs(adjacent,opposite longitude)], with opposite longitude offsetI/2. Then Br=fr/Ar,Bθ=−fs/At,Bφ=fp/Aparea. The minus sign converts northward s flux into southwardθ field. This reproduces the original Output.bc values without shipping unused plotting/interpolation framework code or ghost-cell arrays. Preserve all staggered components: Br(K+1,J,I),Bθ(K,J+1,I),Bφ(K,J,I+1). Output batch dimension is first; the first/last azimuthal faces are physical periodic duplicates.

For physical flux diagnostics, polar-face areas are exactly zero: the adjacent-face surrogate above only defines the regular field value at the coordinate singularity. This distinction is independently exercised with a non-axisymmetric field. The net outward flux per cell is Δ(fr)−Δ(BθAt)+Δ(fp), with the polar minus sign above. It cancels to numerical precision. This conservation test alone would not prove correct physics, so an independent continuum dipole is also checked. For inner Br=s and outer radiusR, analytic Br(r,s)=s[1+2(R/r)³]/[1+2R³] and Bφ=0. A3% relativeL2 discretization bound on a60×120×35 test grid is fixed before assessment. Original-stage/full-field parity uses much tighter numerical tolerances.

## Comparisons, metrics and physical limits

Notebook freshly solves the native benchmark and three source-radius conditionsR=2,2.5,3, then recomputes a separate experiment retaining every second measured longitude. The latter uses actual retained centers1.5°,3.5°,…relative to the original edge; its effective grid edge is22.5°. Periodic interpolation of that newly solved coarse source map onto native centers is labeled an experiment, never a substitute native output. All three components are also averaged to cell centers for a labeled comparison display; these plotting averages never alter saved staggered arrays.

Every full component is evaluated against the independently reproduced original standard result with uncentered cosine similarity and dynamic-range-normalized RMS error. No true coronal array exists. NCC≥.999999 andNRMSE≤1e−6 are nonzero numerical agreement tolerances, justified by independent repeated original equality and eigensystem/backend precision, includingfloat32 submissions. The Judge evaluates all six metrics and accepts only if each passes. There is no alignment, clipping, unit fitting or component substitution.

Figures report signed shell flux, unsigned shell flux, radialRMS and finite-volume cell imbalance from fresh arrays. They explicitly label99th-percentile display limits and show numerical parity residual at its full scale; scientific arrays remain unmodified. The current-free/static/source-surface assumptions omit coronal currents, transient eruptions and solar-wind dynamics. GONG synoptic data combine7391inputs over time and include uncertain polar/line-projection calibration. Agreement with an original solver is not validation of true-corona accuracy.
