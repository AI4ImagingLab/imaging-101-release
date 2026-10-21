# Approach: solid-angle diffusion orientation reconstruction

## 1. Bound inverse problem and acquisition

Recover an order-eight even real spherical-harmonic (SH) representation of the water-displacement orientation density in every voxel of a 24×24×1 real human MRI crop. The deposited Stanford acquisition has 160 magnitude volumes: 10 unweighted references followed by 150 directions at b=2000 s/mm². Image spacing is 2 mm in each spatial axis. No tissue mask, spatial resampling, angular subsampling or noise synthesis is introduced. The output is 45 real coefficients per voxel, not a diffusion tensor, axon fraction or radial propagator.

The data are motion-corrected acquired magnitudes. Diffusion timing, scanner gain and noise variance are unavailable; the approximation does not invent them. Calibration consists only of the measured b0 intensities and deposited gradient directions/bvalues. No hidden anatomy or response function enters reconstruction.

## 2. Continuous physical model and identifiability

Let P(r) be the probability density of displacement vector r, g and u unit directions, q the diffusion wavevector magnitude, S the diffusion-weighted signal and S0 the mean unweighted signal. The Fourier acquisition model and solid-angle orientation density are

```math
E(q\mathbf g)=S(q\mathbf g)/S_0=\int P(\mathbf r)\exp(-2\pi i q\mathbf g\cdot\mathbf r)d^3\mathbf r,
\qquad \Psi(\mathbf u)=\int_0^\infty P(r\mathbf u)r^2dr.
```

The r² Jacobian distinguishes a solid-angle density from an unweighted radial integral. Psi integrates to one on the unit sphere and has sr^-1 units. A single shell does not determine the complete radial Fourier transform. The selected Aganj single-shell constant-solid-angle approximation reduces the desired orientation density to a spherical derivative/Funk-Radon transform of log (-log (E)). This is an approximation to displacement orientation, not evidence of physical axon counts.

There is no spatial convolution: voxels are independent. Spatial boundaries have no extrapolation or periodic convention. Only angular symmetry is imposed: even SH degrees enforce Psi (-u)=Psi (u). Angles use the deposited image-coordinate convention, not an undocumented scanner/world rotation.

## 3. Discrete signal and harmonic conventions

Denote voxel index v, acquisition j, and bvalue b (j). Make a float32 copy of all 160 values, floor each magnitude at 1e-5 a.u., then divide by the float32 mean of samples b<=50 s/mm². Retain samples b>50 and clip their dimensionless attenuations to [0.001,0.999]. Form y (v,j)=log (-log (E(v,j))) using that float32 path. This preserves the original precision and guards the logarithm; it is not a fitted noise model.

For degree l=0,2,4,6,8, enumerate m=-l,...,l, giving K=45 coefficients. Let Y_l^m (theta,phi) be orthonormal complex Condon-Shortley harmonics. Use the actual original default legacy Descoteaux real basis:

```math
B_{lm}=\begin{cases}
\sqrt2\operatorname{Im}Y_l^{|m|},&m>0,\\
Y_l^0,&m=0,\\
\sqrt2\operatorname{Re}Y_l^{|m|},&m<0.
\end{cases}
```

Polar angle theta=acos (gz/||g||), and azimuth phi=atan 2(gy,gx). Division by vector norm removes the deposited gradient text rounding (maximum norm discrepancy 7.72112981e-6). Unweighted zero directions never enter angular construction. B[j,k] is 150×45 float64. The negative-m absolute value is deliberate: replacing it with the nonlegacy paper convention would change the required coefficients even if an angular function were re-expressed consistently.

## 4. Regularized angular fit and analytic inverse

Let L(k)=-l (l+1) be the Laplace-Beltrami eigenvalue and F(k)=P_l (0) the Legendre value at zero. For each voxel solve

```math
\min_a\|Ba-y\|_2^2+\lambda\|\operatorname{diag}(L)a\|_2^2,\qquad\lambda=0.006.
```

There is no sample-count normalization. Differentiating gives (B^T B+lambda diag (L²)) a=B^T y. Rather than square the condition number, form the 195×45 matrix M=[B;sqrt (lambda) diag (L)] and use a=M^+[y;0]. Define H=M^+[:,0:150]. NumPy 2.4.6's generic SVD pseudoinverse uses its verified default rcond = 1e-15: singular values at or below 1e-15 times the largest singular value are discarded; it does not call a domain inverse.

The task-specific CSA spectral inverse is

```math
A=\operatorname{diag}\left(\frac{F L}{8\pi}\right)H,
\qquad c(v)=A\,y(v),\qquad c_{00}=\frac1{2\sqrt\pi}.
```

The product includes the negative Laplacian eigenvalue; no extra minus sign, absolute value or 2 pi factor is added. Because B00=1/(2 sqrt (pi)) and every nonconstant SH integrates to zero, fixing c00 ensures integral (Psi)=1. The reconstructed angular function is Psi (u)=sum_k c (k) B_k (u). Smoothing penalizes the transformed-signal fit, not the final ODF directly. No nonnegativity constraint or coefficient clipping is applied; truncated angular expansions can have negative lobes.

## 5. Complete selected algorithm and numerical policy

1. Load the three public arrays without pickle; remove the batch axis only from bvalues/directions, retaining signal shape (1,24,24,1,160).
2. Execute the exact float32 normalization, shell classification, clipping and double logarithm described above.
3. Construct the weighted-direction basis once, the diagonal penalty and augmented SVD pseudoinverse. Apply the Legendre/Laplacian multiplier to make A(45,150).
4. Multiply all voxel double-log rows by A.T. Mixed float32/float64 multiplication returns float64 coefficients; overwrite the constant mode with the analytic unit-integral value.
5. Save the complete (1,24,24,1,45) real float64 array without masks, scale fits, rotations or spatial changes.

| Selected parameter | Value | Role |
|---|---|---|
| Harmonic maximum degree |8| Even expansion;45 coefficients |
| Angular smoothing |0.006| Unnormalized transformed-signal quadratic penalty |
| Raw magnitude floor |1e-5 a.u.| Original float32 numerical guard |
| Attenuation interval |[0.001,0.999]| Makes both logarithms finite |
| Unweighted cutoff |50 s/mm²|10 acquired baseline volumes |
| Constant coefficient |1/(2 sqrt (pi)) sr^-1| Unit integral on sphere |

This is a closed-form reconstruction after a nonlinear signal transform. There is no initial iterate, learning rate, iterative update, line search, optimizer success flag or convergence threshold. Termination is completion of the SVD and one matrix product, not an artificial iteration count. The CLI performs no plotting and imports no upstream package.

## 6. Comparison groups, evaluation and evidence

The original DIPY 1.12.0 run on these exact observations was independently frozen before local extraction. It supplies baseline_reference.npz; no independent anatomical truth exists. The notebook retains acquired-b0, angular-coefficient-norm and coefficient-error maps plus three angular original/fresh comparisons and a metric table. Historical paper results are not relabeled as measurements of this crop. The existing cleaned snapshot checks repeatability only and remains byte-immutable during review execution.

The standalone metric is||c-c_ref||2/||c_ref||2 across all 25,920 coefficients. No alignment, scale fitting, mean subtraction, magnitude conversion or voxel exclusion occurs. Boundary 1e-6 was fixed before extraction, approximately eight float32 epsilons. Original repeats gave 0 discrepancy; an independent one-ULP upward perturbation of normalized float32 observations gave coefficient sensitivity 1.3130226009468278e-7, making the boundary 7.62 times that sensitivity. Fresh relativeL2 is about 2.885e-15, not a physical accuracy claim.

Frozen parity separately tests normalized signal, B, A and the full output. Independent quadrature verifies SH orthogonality, antipodal symmetry and an isotropic density 1/(4 pi) with unit sphere integral; a dot-product identity verifies the linear sampling adjoint. Known-vector metric and malformed/poor/evaluator-failure Judge tests address a different contract. Passing these establishes numerical extraction and mathematical identities; single-shell modeling, noise, finite angular bandwidth and unavailable timing remain limitations.
