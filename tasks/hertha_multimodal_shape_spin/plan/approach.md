# Implemented conditional multimodal inverse

The selected acquisition is the complete native Hertha `135_oct.ini` example:
30 relative photometric curves/923samples, two150by150 author-reduced AO windows
and matched PSF windows, and18detected occultation chords. The identifier135 is
the asteroid number. All data are real author-supplied reduced observations.
This is a conditional method estimate, not independent asteroid surface truth.
The scientific model follows the2015 ADAM paper; the exact later author source
and its configuration, rather than a claim to reproduce all2015 defaults, fix
the numerical conventions below.

## Representation, coordinates and units

The closed outward triangulation has402vertices/800facets. Its fixed colatitude
and longitude arrays are supplied publicly as `vertex_angles`; faces are supplied
as zero-based `facets`. Native rings use10latitude subdivisions per quadrant:
upper ring index i has colatitude i*pi/20 and4i longitude samples j*pi/(2i),
lower rings mirror through pi/2 in reverse order, with north and south poles.
`sphere_grid` preserves the actual source multiplication order for these angular
steps. It returns the native ordered faces, checked against the original export.

Let Y be the402by100 real spherical-harmonic matrix through degree9, indexed
l(l+1)+m, with l=0..9 and m=-l..l. The associated Legendre primitive includes
the Condon--Shortley sign. Its normalization is

```math
K_{lm}=\sqrt{\frac{(2l+1)(l-|m|)!}{4\pi(l+|m|)!}},\quad
Y_{l0}=K_{l0}P_l^0(\cos\theta),\quad
Y_{lm}=\sqrt{2}K_{lm}P_l^{|m|}(\cos\theta)
\begin{cases}\cos(m\phi)&m>0,\\\sin(|m|\phi)&m<0.\end{cases}
```

Here theta/phi are the fixed angular grid coordinates. SciPy `lpmv` is a general
mathematical primitive, not a shape inverse; the real basis, normalization,
coefficient order and all shape equations are local. The selected validated
library evaluation is retained. An explicit source-style recurrence was also
tested and a complete latitude/longitude gate failure is preserved privately:
algebraic equality does not imply identical stopping points in this nonconvex
problem. It is not presented as another accepted reconstruction.

The300shape coefficients are three100component vectors a,b,c. With exponentials
interpreted as coordinate radii in units1km, coordinates are

```math
x_i=e^{(Ya)_i}\sin\theta_i\cos\phi_i,\quad
y_i=e^{(Y(a+b))_i}\sin\theta_i\sin\phi_i,\quad
z_i=e^{(Y(a+c))_i}\cos\theta_i.
```

Every coordinate is in km. The mesh Jacobian has XYZ-coordinate-major rows and
[a,b,c] coefficient columns: dx/da=diag(x)Y; dy/da=dy/db=diag(y)Y;
dz/da=dz/dc=diag(z)Y; all other blocks vanish. Three identity padding rows/columns
carry the spin derivatives. This is not a finite-difference mesh model.

Internal spin is(colatitude beta,longitude lambda,angular speed omega,phase0),
in radians/radians-per-day. Output is(latitude90-beta in degrees,longitude
modulo360in degrees,period48*pi/omega in hours). The fourth phase is fixed0 at
JD2443846.0; no fitted alignment is performed. Inertial-to-body rotation at
supplied relative time t(days from this epoch; AO/occultation source light-time corrections retained, photometry only JD minus epoch) is M=F(omega*t)B, where

```math
F(q)=\begin{bmatrix}\cos q&\sin q&0\\-\sin q&\cos q&0\\0&0&1\end{bmatrix},\quad
B=\begin{bmatrix}
\cos\beta\cos\lambda&\cos\beta\sin\lambda&-\sin\beta\\
-\sin\lambda&\cos\lambda&0\\
\sin\beta\cos\lambda&\sin\beta\sin\lambda&\cos\beta
\end{bmatrix}.
```

Analytic beta/lambda/omega derivatives differentiate these matrices. Given an
inertial observer unit vector E and camera-up U, the camera rows are z=E/||E||,
x=(U cross z)/||U cross z||, y=(z cross x)/||z cross x||. Thus body-to-camera is
R*M transpose. Positive occultation offsets add to projected coordinates;
positive AO offsets instead multiply the Fourier model by exp(+2*pi*i*f dot s).

## Facet visibility, normals and reflectance

For each triangle(v1,v2,v3), g=(v2-v1)cross(v3-v1), area A=||g||/2, normal n=g/||g||,
and centroid c=(v1+v2+v3)/3. Normal derivatives use the projection
(I-n*n transpose)dg/||g||; area derivatives are n dot dg/2. The C++ kernels contain
the explicit per-coordinate expansions and scatter each contribution to its
three vertices.

The source potential-blocker convention is retained. Triangle k is a candidate
above triangle t when one nonshared vertex w satisfies(w-c_t)dot n_t>0 and
(w-c_t)dot n_k<0. A candidate back-facing to the observer or Sun blocks t when
the centroid line intersects that triangle. The native barycentric determinant
cutoff is1e-6; barycentric coordinates must be nonnegative and sum to at most1.
There is no derivative through a discrete visibility switch. Illumination and
visibility use mu=n dot M E and mu0=n dot M E0, both nonnegative. This preserves
the selected source facet test, rather than silently assuming convexity.

Selected reflectance is constant albedo with the LS/Lambert coefficient0.1.
Disk-integrated intensity is sum over visible facets of
A*mu*mu0*(1/(mu+mu0)+0.1). Each relative lightcurve divides by its own model mean;
the derivative differentiates that mean as well as every facet. There are no
free photometric phase-law parameters or albedo coefficients in this binding.

## Native AO observations and analytic Fourier acquisition

FITS format conversion is private. The public arrays preserve the original
one-based(51,51)150by150windows; local preprocessing applies max(intensity,0),
an even-size rFFT, multiplies bin(i,j) by(-1)^(i+j), divides by the DC real value,
and discards that bin. This places the origin at zero-based(75,75). Frequencies
are j/(150*pixel_pitch) horizontally and NumPy FFT signed row frequencies
vertically, with pixel_pitch0.009942arcsec. All11399remaining samples per exposure
and all matched PSF samples are retained. Four complete original arrays verify
both transforms and frequency order; no fitted PSF is generated.

Let d=206264.80624709636/(distance_AU*149597871.0) arcsec/km and B_t=mu0*(1/(mu+mu0)+0.1).
For projected triangle T_t, the unblurred normalized complex Fourier model is

```math
H(\boldsymbol f)=
\frac{\sum_{t\in\mathcal V} B_t\int_{T_t}
e^{-2\pi\mathrm{i}\boldsymbol f\cdot\boldsymbol u}\,d\boldsymbol u}
{d^2\sum_{t\in\mathcal V}B_t\mu_t A_t}.
```

The numerator is a signed oriented triangle Fourier integral. Local kernels
retain the explicit edge antiderivatives and their six projected-vertex
derivatives, including the native near-parallel edge limit selected when
abs(f dot edge)<=1e-14. Normal, reflectance, projection, total-brightness
normalization and spin derivatives are all included. The exposure prediction is
exp(log_gain)*PSF_FT*exp(+2*pi*i*f dot offset)*H. Log-gain derivative equals this
prediction; offset derivatives are its2*pi*i*fx/fy multiples.

Independent physical validation integrates a triangle directly in barycentric
coordinates. A separate NumPy Judge uses the Green-boundary identity

```math
\int_T e^{-2\pi\mathrm{i}\boldsymbol f\cdot\boldsymbol u}\,d\boldsymbol u
=\frac{\mathrm{i}}{2\pi\|\boldsymbol f\|^2}
\sum_{(a,b)\in\partial T}
\left(f_x(b_y-a_y)-f_y(b_x-a_x)\right)
e^{-\pi\mathrm{i}\boldsymbol f\cdot(a+b)}
\mathrm{sinc}(\boldsymbol f\cdot(b-a)).
```

Here sinc(q)=sin(pi*q)/(pi*q), a/b are consecutive2Dvertices, and i is the
imaginary unit. The nonzero-frequency denominator is safe because DC is removed.
This independently checks the complete physical objective without importing
the reconstruction or compiled kernels. An extreme1e-7cycles/arcsec diagnostic
retains the original edge formula's cancellation error; it is outside the bound
frequency grid and is neither silently corrected nor presented as sampled data.

## Occultation endpoints

The public endpoints are centered by the original common endpoint mean and the
public times already have the distance/c correction with c=299792.458km/s. The
rotation is evaluated at the mean of all36endpoint times. Native camera-up and
observer vectors set the equatorial sky plane. All18native positive flags are
treated as detections; no chord is masked. Chord-time offsets are fixed0seconds,
and no reported timing-error reweighting is introduced.

For a measured line through a,b and projected mesh edge x,y, set
den=(b_x-a_x)(x_y-y_y)-(b_y-a_y)(x_x-y_x). If abs(den)>=1e-10, intersection
fraction q=((a_x-x_x)(b_y-a_y)-(a_y-x_y)(b_x-a_x))/den lies in[0,1]. The point
is x+q(y-x). The two outer intersections along the oriented measured line give
entry/exit. Four residuals are predicted minus measured endpoint coordinates.
The explicit rational intersection derivatives propagate through projection,
spin and vertices. Native fallback distance expressions remain in the local
kernel for missing intersections; the selected complete accepted solutions
intersect every required detected chord. The independent Judge also verifies
those actual intersections.

## Selected residual and regularizers

Parameter order is300shape coefficients,3internal spin quantities,4AO offsets,
2AO log gains,2occultation offsets:311unknowns. There are923LC residuals,
45596AO residuals(all real samples across both exposures, then all imaginary
samples),72occultation coordinates, and4scalar regularizers:46595rows.

Let C_t be the potential-blocker set above; adjacency requires a shared edge.
Let E_mesh be the undirected edges and V the enclosed signed volume. The scalar
regularizers are

```math
R_C=\frac{\sum_t\sum_{k\in C_t,\ k\sim t}A_k(1-\boldsymbol n_t\cdot\boldsymbol n_k)}{\sum_tA_t},\quad
R_O=\sum_{l=0}^9\sum_{m=-l}^l\frac{l}{9}(b_{lm}^2+c_{lm}^2),\quad
R_D=\sum_{(t,k)\in E_{\mathrm{mesh}}}(1-\boldsymbol n_t\cdot\boldsymbol n_k)^2,\quad
R_M=\|\boldsymbol c_{\mathrm{volume}}\|_2.
```

The native constant-albedo selected branch has no center-of-mass dead zone:
its comment about clipping below0.01km applies to another D-matrix branch, not
the actual null-D call. The center is evaluated by the closed-surface first
moment, equivalent to signed tetrahedra with origin as fourth vertex. Its
analytic derivative and the explicit convexity/dihedral derivatives are local.
The source early return made its trailing allocation frees unreachable; the
extraction moves only that return after those frees, preserving every numeric
expression and expression order.

The complete objective is ||r||² with residual vector
[8*(observed relativeLC-model),2*real(AOobserved-model),2*imag(AOobserved-model),
0.1*(OCpredicted-observed),30*R_C,5*R_O,0.5*R_D,1*R_M]. Regularizer residuals
are squared by the outer objective; R_O and R_D must not be silently replaced
by square roots. Weight values define the author's conditional estimator and
are not claimed as calibrated noise variances. Joint Jacobian J=-dr/dp is
positive model derivative for LC/AO and negative residual derivative for
occultation and regularizers. Coordinate-major mesh derivatives are chained
analytically into harmonic parameters.

## Initialization, updates and stopping

The independent starting ellipsoid uses43.56,43.56,35.64km. Native approximate
normalizer0.2821 is retained for the constant coefficients: a0=log(43.56)/0.2821,
b0=0, c0=log(35.64)/0.2821-a0; all other shape coefficients0. Starting spin is
latitude52deg,longitude272deg,period8.40060hours,phase0. AO offsets/log gains
start0; occultation offset starts(0,-10)km. No hidden result initializes any
unknown.

At each accepted state compute N=J transpose J and h=J transpose r. The native
diagonal-damped additive update is

```math
\left(N+\lambda\operatorname{diag}(N)\right)\Delta p=h,\qquad
p_{\mathrm{trial}}=p+\Delta p.
```

N is311by311. SciPy's general symmetric solve uses lower=True, matching the
native LAPACKE lower-triangle convention. Lambda starts0.1. A strict decrease
accepts the entire joint update and divides lambda by10; otherwise multiply
lambda by10 and reuse the current linearization and right-hand side. There is
no field clamp, per-block update, generic optimizer replacement, or fitted
reference alignment. Accepted states recompute all forward operators,
visibility, normalizations, derivatives and penalties. Stop after two consecutive
accepted decreases below0.1, lambda>1e6, or100total trials. The decrease counters
start10; rejections do not replace accepted-decrease history. There are no new
synthetic observations or noise perturbations, so no generator is shipped.

## Evaluation and scientific comparisons

Every required output is evaluated. Each XYZcoordinate uses its own reference
dynamic range; lat/lon/period use their physical units with longitude wrapping;
AO offsets/log gains and occultation offset use block RMS. No mesh or spin
alignment is allowed. The original accepted-state field, rather than the stale
native outshape snapshot written before final remeshing, is the method reference.
The exact field and physical-objective gates were fixed using independent
complete original O0/O1runs before local inversion. Actual truth would take
precedence if available, and invalid truth never silently falls back.

The notebook retains independent original O1/O0comparisons and the author's
native subdivision example as explicitly labeled original runs. The latter uses
the author subdivision configuration with different stated weights and support,
so it is a shape-support reliability comparison, not an isolated same-weight
ablation. Fresh results always come from the full local selected method. Paired
data/model residuals, every photometric curve, projected chords, numerical
backends and forward spin/shape sensitivity show both fit quality and the
remaining conditional uncertainty.


For a detected chord missed by a trial silhouette, retain the source support-line sigmoid optimization penalty, not a fictitious endpoint. With unit chord normal n, signed vertex distances are d=(a-v) dot n. If both signs occur, use the negative extreme of smaller magnitude; otherwise use the smallest absolute nonzero distance (source empty-side sentinel 10^6 km). Set s=1/(1+exp(-8d)), where 8 has units km^-1. The four unweighted residuals are [a_x(1+s),a_y,b_x,b_y], multiplied by the same occultation residual weight. Independent Judge details mark unavailable intersections NaN and count missed chords; scalar objective remains finite. All bound accepted fits have real intersections; parameter sensitivity retains this native fallback. Original GCC primitive fixtures cover above/below/intersecting/vertical support lines.
