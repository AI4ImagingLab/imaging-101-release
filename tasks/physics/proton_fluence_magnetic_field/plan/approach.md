# Implemented method

## Physical coordinates and supplied constants

The two image-array axes are x then y, each with200 pixel centres from−0.0995 to0.0995cm and nominal spacing0.001cm. Actual spacing is subtracted from the supplied coordinate arrays, preserving their floating-point representation. In the laboratory right-handed frame X points right, Y up and Z out of the detector. Protons propagate along−Z. The author's phrase “z into the screen” names the beam direction; that direction and the displayed x/y axes form a left-handed triple. It must not be substituted for positive laboratory Z in a component cross product.

The supplied constants are ri=1.3cm, li=0.2cm, rs=20cm, v=5.24e9cm/s, mp=1.67e−24g, q=4.803e−10statC, c=3.0e10cm/s. They are preserved, including the rounded speed. Define the dimensionless geometric magnification

```math
M=\frac{r_s+r_i+l_i/2}{r_i+l_i/2}.
```

With a positive proton charge, the small-angle transverse velocity impulse w, in cm/s, and path-integrated transverse magnetic field b, in G cm, satisfy

```math
\mathbf w=\frac{q}{m_pc}\widehat{\mathbf k}\times\mathbf b,\qquad
\widehat{\mathbf k}=-\widehat{\mathbf Z},\qquad
b_x=-\frac{m_pc}{q}w_y,\quad b_y=\frac{m_pc}{q}w_x.
```

This rotation is checked with an independent laboratory-basis Lorentz ODE for known positive Bx. No sign is fitted against truth. The source README repeats wBx in its Bxpath sentence, but its executed `magpath` function uses wBy. The executed signs are physically consistent with the stated into-screen display convention. The positive-Z laboratory interpretation initially written during cleaning was incorrect and is explicitly corrected in validation evidence.

## Forward model and inverse unknown

The demagnified detector map T has units cm. It is the gradient of a mapping potential Phi in cm². The native approximation is irrotational transverse transport, positive map Jacobian, uniform incident illumination and zero normal boundary deflection:

```math
\mathbf T=\nabla\Phi=\mathbf x+\frac{r_s}{Mv}\mathbf w,\qquad
f(\mathbf T)\det\nabla^2\Phi=f_0,\qquad
\mathbf T\cdot\mathbf n=\mathbf x\cdot\mathbf n.
```

Here f is observed counts per pixel, f0 is a same-shaped constant array equal to the float64 mean of f, and n is an outward unit boundary normal. Counts per equal-area demagnified pixels can be used because the common pixel area cancels. No incident target-derived image is provided or used. The selected spatial unknown ultimately contains both b components on all200² pixels.

## Discretization and original update

The initialization is Phi=(x²+y²)/2, the identity map. On each iteration, `np.gradient(Phi,dx,dy)` supplies centered two-point first derivatives inside the grid and first-order one-sided differences on edges. Replace the x-gradient on the first/last x row with the corresponding x coordinate. Replace the y-gradient on the first/last y column with the corresponding y coordinate. These are the source's normal Neumann conditions.

Differentiate those adjusted gradients again: Hxx=Dx(Tx), Hxy=Dy(Tx), Hyy=Dy(Ty). Set Hxy to zero on all four edges. The source uses this Hxy twice in the determinant rather than an independently computed Dy(Tx)*Dx(Ty) product. This exact discrete convention is retained and covered by rectangular, nonzero-mixed-derivative upstream fixtures.

```math
J=H_{xx}H_{yy}-H_{xy}^{2},\qquad
F=\log\left(\frac{f(\mathbf T)|J|}{f_0}\right),\qquad
\Phi^{k+1}=\Phi^k+\Delta\tau F^k.
```

All operations are float64. The observed f is interpolated bilinearly by a general SciPy regular-grid interpolator, using mean(f) outside the coordinate rectangle. The relaxation time increment has units cm², since F is dimensionless and Phi has units cm². The default increment is1e−8cm². This is a log-parabolic Monge–Ampère relaxation; no loss minimizer or different inverse algorithm replaces the update. There is no explicit penalty, line search, filtering or data rescaling. Finite iteration time and the potential-map assumption constrain the reconstruction.

The paper describes a positive determinant; source code evaluates its absolute value. The source convention is preserved, and the selected solution's positive Hessian eigenvalues and determinant are checked independently. The fresh solver also counts nonpositive determinants during the entire trajectory. Negative determinants are not silently clipped or repaired.

After computing F, form the new potential and record RMS(Phi_new−Phi). Then stop if the Euclidean norm of F is below0.001; otherwise continue up to400000 updates. The native exact-data run reaches the maximum, with norm(F)=0.3668422103144939, rather than the requested tolerance. Its RMS log-mass residual is0.0018342110515724693. Neither the notebook nor provenance calls this tolerance convergence.

The returned potential has received the final Euler update, while its returned map and Hessian are from immediately before that update, exactly as the original function returns them. The magnetic field is computed from that returned map. Recomputing its gradient from the final potential would change the selected source convention and is not done.

## Physical conversion and analysis comparisons

The recovered velocity components are M(v/rs)(Tx−x) and M(v/rs)(Ty−y). Convert them with the Lorentz rotation above, preserving the source's multiplication order and rounded constants. The model does not add a relativistic gamma multiplier. The existing author's count generator is unavailable, so the particle-binned forward prediction from the supplied actual field is explicitly treated as an approximate-model check, not a claimed exact reproduction of its original noise process.

The notebook computes a separate weak-contrast Poisson comparison. Write Phi=(x²+y²)/2+u, retain first-order contrast terms and solve

```math
\nabla^2u=1-f/\overline f.
```

For this comparison only, the cell-centred homogeneous-Neumann Laplacian is diagonalized with an orthonormal DCT-II. Its eigenvalues are −4 sin²(pi*j/(2*Nx))/dx²−4 sin²(pi*l/(2*Ny))/dy², with j,l the nonnegative spatial mode indices. Set the zero-mode solution to zero to fix the additive gauge. Convert the resulting displacement through the same physical constants. This is a fresh computed baseline, not the nonlinear task answer.

An additional analysis helper uniformly samples the incident plane with n² equal-weight rays per pixel, bilinearly interpolates a supplied physical field and bins the deflected coordinates. The bound beam direction is−1 along laboratory Z; +1 is an explicit incorrect-sign control. Each ray carries mean(f)/n² counts. Rays are streamed in128-row blocks to limit memory. The exterior field is zero for this finite-object analysis helper. This discretization is independent of the inverse's differential Jacobian residual. Its finite-grid/approximate-physics discrepancy is reported, not concealed with fitted gain or a mask.

## Evaluation

The existing author Bxpath/Bypath arrays are actual simulated field truth, separately archived from observations. They are unchanged and never used in reconstruction or initialization. Each component is scored on every pixel with cosine NCC and RMS error divided by that component's true dynamic range. No shift, crop, sign change, gain fit or implicit unit conversion is permitted. Quality boundaries were frozen before local extraction at0.9 times the corresponding original NCC and1.1 times its dynamic-range NRMSE. Independent stage and full-field parity have separate, much tighter numerical tolerances; passing the Judge does not establish parity.
