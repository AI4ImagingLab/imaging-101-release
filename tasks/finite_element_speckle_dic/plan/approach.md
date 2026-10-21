# Finite-element displacement imaging: implemented derivation

## 1. Physical unknown and bound discretization

The acquisition is brightness transport between two unnormalized 8-bit speckle
images, held exactly in float64. Coordinates are pixels, x right/y up, origin at
the lower-left pixel center. Images have already been vertically reversed and
cropped as recorded in validation. The supplied regular reference mesh has 15
nodes and eight 80-pixel Q4 elements. No length or time conversion is made.

For parent coordinates xi,eta in [0,1], N=[(1-xi)(1-eta),xi(1-eta),xi*eta,
(1-xi)*eta]. Reference pixel location X=N Xnod and current location x=N xnod.
Here Xnod and xnod are four-by-two nodal arrays, and displacement d=xnod-Xnod.
The inverse reference mapping is affine because the supplied elements are
axis-aligned rectangles. This specialization is algebraically equal to the
general upstream Q4 inverse on the bound undeformed mesh, not an alternate
image-motion model.

## 2. Photometric objective and exact discrete counting

The objective is the unweighted, unnormalized sum R=sum_e sum_p
[I0(Xp)-I1(Np xnod_e)]^2 in squared digital counts. Each integer pixel within an
element's inclusive bounding rectangle is integrated with unit weight. Shared
element-edge pixels are deliberately counted for each incident element, as in
upstream; there is no Gauss quadrature, de-duplication or division by pixel count.
No regularizer or intensity normalization is added. Continuity comes from shared
degrees of freedom, not a strain penalty.

## 3. Reference linearization, assembly and transpose

Reference gradients are centered finite differences gx=[I0(y,x+1)-I0(y,x-1)]/2
and gy=[I0(y+1,x)-I0(y-1,x)]/2 in digital counts per pixel. For each element B has
eight rows: gx*N transposed followed by gy*N transposed. Thus B is J transpose,
where J maps local nodal increments ordered xxxx/yyyy to intensity increments.
Assemble H=sum_e P_e^T B_e B_e^T P_e, with P mapping global interleaved x/y
degrees of freedom into this local order. Store H inverse explicitly, preserving
the published/reference implementation. H is positive definite on the bound
textured data; singular texture should fail rather than receive an undocumented
regularizer.

## 4. Intensity interpolation and boundary rules

The deformed full image is spline-prefiltered once by scipy.ndimage.spline_filter
with order=4 and its default mirror boundary. Then each initial element bounding
box is extracted with pad=10: rows ymin-pad:ymax+pad and columns xmin-pad:xmax+pad
(upper endpoint excluded). Its interpolator is map_coordinates(order=4,
prefilter=False, default constant-zero exterior). Evaluate each current coordinate
relative to that unchanged cropped frame. All bound mapped pixels remain inside
the valid crop. The order-four setting follows repository quick_start rather than
silently replacing it by the cubic/quintic alternatives discussed in the paper.

## 5. Modified Newton iteration

Initialize xnod=Xnod. At each update, r=I0-I1(current), C=sum_e P_e^T B_e r_e,
delta=H^-1 C; add interleaved delta to xnod. The plus sign follows linearizing
I1(x+delta)=I1(x)+J delta. This is the paper's fixed-reference modified Newton
method, not an exact current-image Hessian. Use unit steps, no line search,
max_iterations=80 and tolerance=1e-8 pixels on max absolute component of delta.
Failure to reach tolerance raises RuntimeError. The bound pair converges after
10 updates. History records pre-update SSD and increment infinity norm.

## 6. Derivation, actual stopping and numerical choices

Let q in R^30 be global nodal increments with ordering [ux0,uy0,ux1,uy1,...].
For element e define gather T_e such that T_e q stacks its four x increments
followed by its four y increments. The pixel Jacobian is J_e=B_e^T T_e, so the
linearized current intensity is I1(x)+J_e q. With r_e=I0-I1, the increment solves
min_q sum_e ||r_e-J_e q||^2. Differentiation gives

```math
\nabla_q R_{lin}=-2\sum_e J_e^T r_e+2\sum_e J_e^TJ_e q,
\quad H=\sum_e T_e^TB_eB_e^TT_e,
\quad Hq=\sum_e T_e^TB_e r_e.
```

The factor2 cancels; no additional half or pixel-count denominator appears.
The local transpose B_e sums intensity perturbations weighted by the reference
gradient and shape values; T_e transpose scatters them into shared DOFs. Assembly
therefore retains both xy cross-blocks rather than solving components separately.
Each unit-weight 81-by81 element contributes6561 samples;8 elements contribute
52488 residual entries including repeated shared edges. Each B is8-by6561 and
H is30-by30. The full sampled intensity residual remains nonlinear; substituting
the fixed reference gradient is the selected approximate Newton step, not an
assertion that it is the exact current-warp gradient. The independent analytic
polynomial finite-difference test verifies the reference Jacobian itself.

| Quantity | Shipped value | Meaning |
|---|---|---|
| Initial nodes | exact public nodes | Zero initial displacement; all nodes free |
| Difference stencil | centered +/-1 pixel | Count/pixel; no periodic stencil |
| Interpolation | order4; prefilter once | Source quick_start convention |
| Element pad |10 pixels | Fixed crops around initial mesh |
| Step multiplier |1 | No line search or damping |
| Maximum updates |80 | Exceeding limit raises, never silently accepts |
| Convergence | max(abs(delta))<1e-8 pixels | Tested after applying an update |
| Arithmetic |float64 | No checkpoint or hidden initializer |
| H implementation | explicit inverse | Source convention, not a pseudoinverse |

History[k,0] is SSD before update k and history[k,1] is that update's maximum
absolute component; the reported final SSD in history is not recomputed after
the last increment. The notebook independently recomputes final brightness RMSE.
A singular normal matrix propagates NumPy LinAlgError. Exhausted iterations raise
RuntimeError. There is no change in objective or artificial iteration to obtain
parity. The bound run takes10 updates; a tighter oracle tolerance changes the
answer by only5.12358155901893e-10 pixels. Nonconvergence and texture degeneracy
are distinct from the absence of independent experimental truth.

## 7. Submission metrics and independent evidence

Only displacement is a required inverse output, shape (1,15,2), real float32/64,
in pixels; no gauge alignment or scale correction. Judge compares every component
to an original-method reference, never to invented truth. RMSE=sqrt(mean((d-ref)^2))
has a predeclared nonzero 1e-5-pixel boundary. NCC is uncentered cosine similarity;
diagnostic NRMSE divides this RMSE by max(ref)-min(ref). No thresholds are inferred
mechanically from a zero self-comparison error.

## 8. Derived strain and limits

For physical review, element-center derivatives D=dN/d(xi,eta) give
F=(xnod^T D)(Xnod^T D)^-1 and Green-Lagrange strain E=(F^T F-I)/2. F and E are
dimensionless. This derived visualization is not an additional independently
measured unknown or ground truth. Analytic rigid rotations must yield E=0.
The standard F convention is stated explicitly instead of blindly reproducing
typographic ambiguity in paper Eq.9 or relying on upstream postprocessing tensor
storage. No Hencky-strain replication or paper full-series figure claim is made.
