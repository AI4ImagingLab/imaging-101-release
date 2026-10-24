# Implemented plane-stress elasticity tomography

This plan describes the shipped explicit implementation. Acquisition and qualification decisions were fixed before extraction; independent upstream fixtures and numerical tolerances are identified in test/validation.md.

All solver and prior formulas below operate on dimensionless numerical coordinates xi=x/L0 and numerical modulus e=E/E0, with displacement u/L0 and force f/F0. In these formulas E denotes the numerical modulus e for consistency with source variable names. Thus R differentiates with respect to xi, d and beta are numerical modulus and squared numerical modulus scales, and alpha is an inverse numerical modulus scale. The source's beta=.001*d^2 is added to squared numerical gradients, not to a dimensional E0/L0 gradient. Numerical constants are unchanged. The physical-unit mapping remains E0=F0/L0^2.

Let g be the 441 by 2 node coordinates, T the 800 by 3 zero-based triangle indices, E the 800 positive element moduli, u the 882 interleaved x/y nodal displacements, and f the corresponding applied nodal force vector. The arbitrary physical units obey E0=F0/L0^2. Lengths and displacement use L0, force F0, modulus E0; Poisson ratio nu=.35 and boundary penalty multiplier c=100000 are dimensionless. Thickness h=.05 L0. Each element has constant plane-stress constitutive tensor

```math
D(E)=\frac{E}{1-\nu^2}\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&(1-\nu)/2\end{bmatrix}.
```

For a triangle of area A, derive the usual affine basis gradients from its coordinates. B maps its six interleaved nodal degrees of freedom to engineering strain (epsilon_xx,epsilon_yy,2epsilon_xy). Assemble K_e=A*h*B^T*D(E_e)*B into global K. On every supplied constrained degree of freedom, replace the diagonal by c*max(diag(K)); retain the off-diagonal terms. This is the author's finite-penalty boundary convention, not elimination or an exact zero-displacement constraint. Solve K(E)u=f. The maximum diagonal depends on E and must be recomputed for every forward call, including perturbed Jacobian states. Data are u_m=U(E)+noise at all nodes, with the source noise consisting of an independent relative .001|u| Gaussian term and an independent .001 range(u_after_first_noise) Gaussian term. Original Octave draws are private; do not publish a seed or target definition.

Qualified weights are derived only from public data: C=diag(2*((.001 range(u_m))^2+(.01 abs(u_m))^2)), W=C^-1 and L=diag(sqrt(diag(W))). The .01 relative coefficient is a source ad-hoc reconstruction choice; it is not the actual .001 relative generating noise. The residual cost is .5*(u_m-U(E))^T W (u_m-U(E)).

Initialization searches the original 200 homogeneous candidates 5+1.25*k for k=0..199. Evaluate candidates k=1..199, retain the original search subset k=9..199, and choose the true minimizing candidate (correct slice offset). Do not extend this range after seeing the solution. Let m be that scalar. Set d=2*m/.5, alpha=-log(.5)/d, beta=.001*d^2. Generate the Delaunay triangulation of element incenters using Qhull options `Qt Qbb Qc`; Qz changes 692/1521 faces and is therefore excluded. Derive the piecewise affine gradient matrices R_x,R_y from those triangles. Preserve the author's deliberate area weights all equal to one. Stabilized TV is sum sqrt((R_x E)^2+(R_y E)^2+beta). Its gradient and Hessian follow direct differentiation and were independently checked before extraction.

The weighted Gaussian prior has mean m*1, variance sigma^2=((4*m-50)/6)^2, numerical correlation length ell=1 (physical length L0), nugget .001 sigma^2, and covariance C_ij=.999 sigma^2 exp(-distance_ij^2/(2*b^2))+.001 sigma^2 delta_ij, with b=sqrt(-ell^2/(2*log(.01))). Distances here use xi coordinates. Its quadratic cost is .5*(E-m)^T C^-1(E-m), gradient C^-1(E-m), Hessian C^-1. Tikhonov uses .5*lambda*E^T E with numerical lambda=1e-5, gradient lambda E and Hessian lambda I; E is the dimensionless numerical modulus defined above.

Use minimum numerical modulus a=.1 and upper threshold b=300. The fixed quadratic constraint cost is k_low*(E-a)^2 below a plus k_high*(E-b)^2 above b, zero inside. With Phi0 the initial total objective, Q=.0001 Phi0, k_low=Q/(.01*a)^2 and k_high=Q/b^2. Keep these coefficients fixed throughout the run, as separately approved objective-consistency qualification. Gradients are twice coefficient times violation, diagonal Hessians twice coefficient on the violated side. Initialize coefficients after evaluating Phi0; the homogeneous initial field lies inside both thresholds. Project trial and accepted parameters below .1 to .1; there is no upper clipping. Here Phi0 is an objective value; F0 remains exclusively the reference force unit.

At each GN step form the central Jacobian J[:,j]=(U(E+dx*e_j)-U(E-dx*e_j))/(2*dx), dx=(machine_epsilon/2)^(1/3) E0, evaluated at the full current E. Solve (J^T W J+prior_H+barrier_H)*delta=J^T W*(u_m-U(E))-prior_gradient-barrier_gradient. Custom source line search first tests 100 steps 1e-10+k*S0, k=0..99, S0=.02 for TV/Gaussian or 1 for L2. Choose the minimum; start refinement at max(1e-10,s_best-S0), then increment S0/(100/3) before each of 100 tests. Choose the minimum refined step (the coarse minimum is not silently added to this second grid). Project each tested field before scoring. Source min-tie behavior must be investigated if encountered; do not invent a new tie rule on a real tie. Accept the refined step and evaluate the same fixed objective. Stop at relative absolute cost change below 1e-6, any cost increase, or source maximum index100 (99 updates). Preserve the increase-triggered last field rather than silently selecting an earlier best field.

Independent checks required: full stiffness symmetry/positive strain energy, homogeneous u proportional to 1/E including penalty replacement, full-model equilibrium directional derivative including the E-dependent penalty, Jacobian column/parity with cancellation-aware tolerances, all prior and barrier gradients/Hessian actions, complete GN direction, exact line-search grids/projection, and complete source trajectory. Actual simulation truth supports ordinary cosine NCC and dynamic-range NRMSE; reference agreement is not substituted for physical truth. An independent forward observation metric must respect the ad-hoc weighting caveat.


## Bound physical acquisition and reference semantics

The public nodal load is0.00275 F0 per right-edge x component, selected through
the author's configurable force input. It was independently simulated and
actually run upstream, rather than produced by scaling a reconstruction. The
original2.75-load rectangular case gave maximum principal strain2.439 and
failed small-strain applicability. The final true source simulation gives
maximum absolute principal strain0.002439117202456555, below the fixed.01
ceiling. Neither original force nor modulus has a documented SI calibration;
E0=F0/L0^2 is a consistent reference-scale convention. No GPa, N or metres are
inferred from the source's arbitrary numbers.

All public arrays preserve float64 observations and explicit zero-based
integer connectivity. Actual simulated E is hidden ground truth. Separate
original and qualified reconstructions are method results, never truth.
Judge NCC is cosine similarity without mean subtraction; NRMSE is RMS
element error divided by the actual reference range. Every one of800 elements
is used, with no alignment or rescaling. A separate independent FEM compares
predicted and public noisy displacements by relative L2 error and computes
maximum absolute principal affine strain. Quality thresholds derive from the
complete qualified TV method against actual truth; full-pipeline numerical
parity has a different independently controlled-backend tolerance.

The literal.001 generating coefficients are retained. The paper's1% example
is historical context. Source ad-hoc relative weight.01 is retained as a
reconstruction setting and computed from noisy observations, not interpreted
as the true covariance. Thus reference residuals include regularization bias
and are not claimed to give chi-squared per degree of freedom near one.

## Numerical implementation details

Precompute each unit-modulus element stiffness and sparse assembly indices
from public geometry; E-linearity permits multiplying these by the current E.
Sparse CSC solves use SciPy's general linear-system primitive. Initialization
uses exact homogeneous scaling U(c*1)=U(1)/c to evaluate the original candidate
grid with one forward solve; the selected candidate is upstream-validated.
All perturbed heterogeneous Jacobian states are explicitly solved individually.
Spatial derivatives use the source-compatible dual Delaunay geometry, not
SciPy's different Qz default. Gaussian covariance is explicitly constructed
then inverted using a general dense linear-system primitive. Prior gradients,
Hessians, barriers and GN updates are local formulas. There is no domain
reconstruction library or upstream import.

TV is the standard entry-point method. The notebook actively computes the
same full path, plus Gaussian and L2 comparison paths and TV multiplier
sensitivity0.5/1/2; those are labeled separately from source results. All
complete paths use maximum index100 and their own actual stopping rule.
Exact line-search ties are an explicit error requiring investigation, not a
silently invented first-minimum convention; no bound source path encountered
such a tie. The original search's unused50-iteration/fno interface arguments
are omitted locally while the actual100+100 search points are preserved.
