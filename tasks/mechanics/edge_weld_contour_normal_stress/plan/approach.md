# Implemented contour restoration inverse

## Unknown, observations and physical convention

The required quantity is the complete3,166-vector q of normal restoration stress sigma(zz) on the supplied cut-face node order. Units are MPa, positive for tension in the fixed author's elastic restoration model. It is not the complete original residual tensor inside the specimen. The signed physical pre-cut release-axis calibration is not independently documented; no externally calibrated sign or original interior tensor is inferred. The standard-method result supplies a hidden reference, not actual stress truth.

Inputs are2,010finite registered and averaged CMM samples (x,y,h) in mm, the original1,917-point fitting selection, a128-point outline,34,826geometry nodes,30,370eight-node bricks,3,166cut nodes and two rigid-gauge anchors. The original two-face registration/averaging is an already completed part of this author's observation record; it is not reexecuted or claimed as a new measurement. The93flagged fitting points remain in public data. Every cut node, including the extrapolated cut-completion region, remains in the output and evaluation.

Let u be Cartesian restoration displacement in mm. The body occupies the supplied full half-beam geometry; its cut plane is model z=1mm and outward normal is minus the z unit vector. Uniform isotropic Young's modulus E=195600MPa and Poisson's ratio nu=.294 are supplied model values. The remaining exterior surface is traction-free. On the cut face only normal traction is unknown; tangential traction is zero. Formally, normal traction minus q times the z unit vector produces an elastic displacement whose normal trace is observed as h. The reconstruction inverts this elastic compliance by fitting h and solving the equivalent normal-displacement restoration problem.

The continuum model is

```math
\nabla\cdot\sigma=0,\qquad
\epsilon=\tfrac12[\nabla u+(\nabla u)^T],\qquad
\sigma=\lambda\operatorname{tr}(\epsilon)I+2\mu\epsilon,
```

where epsilon is infinitesimal strain, sigma is Cauchy stress, I is the identity, lambda=E nu/[(1+nu)(1-2nu)] and mu=E/[2(1+nu)] are Lame moduli. A positive sigma(zz) corresponds to negative-z outward traction at the cut. The observation convention is specifically restoration displacement u(z)=h, not an independently verified raw release displacement. This convention is public and fixed.

## Frames and numerical precision

The author FE generator adds1mm to each geometric coordinate. The observed height remains a displacement; it is not shifted. Spline queries use the supplied pre-deck mesh x/y coordinates minus1mm. The original VTK point container stores evaluated heights as float32, which are then written to six decimal places in millimetres. `boundary_displacements` performs those two conversions explicitly before applying constraints.

The supplied mesh retains its full author pre-deck coordinate precision. FE assembly uses those coordinates written and parsed with the original six-decimal-mm representation. Keeping only rounded input geometry would lose the separate surface-evaluation calibration. Both roles are stated in README. No inferred stress, target-dependent geometry, synthetic object or reconstruction initializes the solver.

## Surface basis and least-squares objective

For reliable samples indexed by V, fit a tensor-product cubic spline. With coefficient c(ab), cubic basis T(a),T(b) and coordinates x(i),y(i), minimize the unweighted sum

```math
\sum_{i\in V}\left[\sum_{ab}c_{ab}T_a(x_i)T_b(y_i)-h_i\right]^2.
```

There is no averaging normalization, additional penalty, target weight, automatic outlier rejection or observation rescaling. Smoothness comes from the fixed finite knot basis. General SciPy FITPACK `bisplrep(task=-1)` solves the fixed-knot least-squares problem; it does not supply the elastic acquisition or inversion.

Default nominal knot controls are3mm in author x and7mm in author y, degrees3/3. For each dimension the code takes n=int((upper-lower)/control), generates n uniformly spaced points including both bounds, inserts three extra lower knots at the start and three extra upper knots before the last knot. Thus endpoints have multiplicity four. Interior spacing is (upper-lower)/(n-1), not necessarily the nominal control value. This paper/code convention is preserved, not silently replaced by a uniform arange with step3or7. Fewer than three generated points per dimension is invalid. All spline variables, fitted coefficients and BC precision are checked against independently captured native fixtures.

## Explicit fully integrated brick physics

Natural coordinates r=(xi,eta,zeta) range from minus1to1. The8node signs are, in element connectivity order: (---),(+--),(++-),(-+-),(--+),(+-+),(+++),(-++). For node i with signs s(i,a), shape function N(i) is

```math
N_i(r)=\tfrac18\prod_{a=1}^{3}[1+s_{ia}r_a].
```

Differentiate this formula directly. The mapping Jacobian J(a,b)=sum(i)x(i,a)dN(i)/dr(b) uses physical node coordinates x in mm; physical derivatives are the natural row derivatives multiplied by inverse J. All determinants must be finite and positive. All eight integration weights are1. The actual native Gauss order is (---),(+--),(-+-),(++-),(--+),(+-+),(-++),(+++), each divided by sqrt(3).

The24element displacement entries are node-major x/y/z. B maps them to engineering strain (epsilon(xx),epsilon(yy),epsilon(zz),gamma(xy),gamma(xz),gamma(yz)); gamma is twice tensor shear strain. The6by6constitutive matrix D has normal diagonal lambda+2mu, normal off-diagonal lambda and shear diagonal mu. Stress order is (xx,yy,zz,xy,xz,yz), matching native captured output. General NumPy tensor products evaluate

```math
K_e=\sum_{g=1}^{8}\det J_g\,B_g^TDB_g.
```

The local assembly explicitly maps each24by24block to the global104,478-degree-of-freedom sparse matrix. K has units N/mm, displacement mm, reactions N and stress MPa=N/mm². There is no reduced mesh, incompatible element, domain library or external FE executable in this runtime. Native CalculiX is used only in the isolated upstream oracle.

## Boundary constraints and direct inverse

Fix normal displacement at every3,166cut node to its evaluated/serialized height. The first anchor fixes x/y displacement to zero and the second fixes y to zero. These three in-plane constraints fix rigid translations and rotation without adding a physical load. The original source's malformed second-anchor card `node,2,0` is ignored by the tested native backend, leaving a rigid rotational gauge. An independent full native run with intended `node,2,2,0` produced relative1.77e-10change in all six integration stress fields (maximum1e-5MPa at native text precision). The local solver fixes the intended gauge explicitly; it never aligns scored stresses or claims the original unconstrained displacement gauge as physical truth.

For free degrees f and prescribed degrees d, eliminate the latter:

```math
K_{ff}u_f=-K_{fd}u_d.
```

General sparse LU uses `splu(permc_spec='MMD_AT_PLUS_A', options={'SymmetricMode':True})`. This is a linear direct solve, with no artificial iterations, optimizer initialization, line search or stopping tolerance. The surface least-squares and elastic system determine the result from public observations/geometry alone. Reactions are K u and strain energy is one half u-transpose K u. All free-force residuals and energy are fresh measurements from the same solve.

## Stress and disclosed source correction

At every element/Gauss point compute sigma(g)=D B(g)u(e). Let G(g,i)=N(i)(r(g)). For each element, the nodal stress vector is inverse(G) times its eight Gauss values. Average contributions to a shared node with equal element counts, preserving the source nodal convention. Select normal component zz at every declared cut node. Internal integration and nodal stress fields remain diagnostics of restoration, not the original interior residual field.

The source1.1.8postprocessor uses a forward shape matrix instead of inverse interpolation, with a different row ordering. Independently prescribed bilinear displacement u(z)=.001xz on a unit cube gives a72.008%relative error in its nodal normal stress. Correct inverse interpolation reduces that to2.53e-7at the native printed precision. Six native controls cover rigid translation/rotation, constant strain, two bilinear fields and a general trilinear field; all six stress components agree with independent analytic derivatives. This narrow postprocessor qualification is explicitly retained alongside the deficient original comparison; it is not described as unchanged source reproduction. Its complete normal-stress field differs from the original deficient output by13.08%relative norm.

## Metrics and fixed acceptance

Required output is real float64(1,3166), MPa, in exactly public surface-node order. Every value is scored without cropping, scaling, absolute values or registration. NCC is uncentered cosine; NRMSE is RMS error divided by the reference's full dynamic range; maximum absolute error is in MPa. Truth has evaluator precedence if independently available, but this real binding has only a qualified method reference.

Before any local implementation, fixed limits were NCC>=.999999, range-NRMSE<=1e-6and max error<=.001MPa. The original native DAT has six scientific decimals. Propagating its half-ULP envelope through the independently validated inverse interpolation and node averaging gives max.0002598076MPa and range-RMS2.37827e-7. Both error limits exceed three times their corresponding independently computed envelope. Complete native repeats are bitwise identical; those zero self-errors are not the tolerance rationale. Zero, best-constant and sign-reversed fields have range errors.17521,.17520and.35042, respectively, and fail. No boundary was loosened after local measurements.

The first local full run has max.000135244MPa and range-NRMSE3.95327e-8, well within the preexisting limits. Known analytic metrics are tested independently against the standalone evaluator and visualization metrics. Native full constitutive/force/equilibrium checks and unit physical controls establish model scale; reference agreement alone does not establish actual pre-cut stress accuracy.

## Active review experiments

The notebook actively computes the bound3/7control case,3/6and3/8nominal-knot variants, and a complete all-points fit that deliberately retains the original cut artefact. Every variant executes the complete mesh/elastic reconstruction; no field is filled from a cached result. The unmasked artefact control produces enormous stresses outside the physical linear-elastic regime and is a failed control, not a valid alternative estimate. All its values remain visible; reliable-mask variants also have a separate profile scale. It preserves the deficient native versus qualified native versus fresh comparison, matched null metrics, full Gauss stresses, observed/fitted height residuals, common-coordinate profiles and strain energy. Figures distinguish their physical scales, known quality selection and absence of independent spatial truth. Observation generation is not needed: all variants reuse the original observed values with declared fitting choices.
