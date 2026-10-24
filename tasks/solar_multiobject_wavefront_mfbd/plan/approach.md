# Implemented Approach

## 1. Bound scientific problem and selected method

The benchmark retains the complete real SST/CRISP burst: one scan, two simultaneous passbands, twelve exposure pairs, and 512 by 512 pixels. It reconstructs a separate latent intensity for each passband while sharing the atmospheric phase within each exposure and isoplanatic region. No trained inverse, archived reconstruction, prescribed blur kernel, or synthetic target participates in reconstruction.

The selected method is the wavefront/KL **joint** likelihood implementation in `torchmfbd` at the exact revision recorded in `test/validation.md`, using `reproducibility/qs_8542/qs_8542_joint.py` and its YAML. In this terminology "joint" contrasts with marginalizing the latent object statistically: the object is eliminated analytically from a joint objective. The shipped method does not implement the repository's marginal PSD fitting, NMF PSF dictionary, VAE representation, temporal regularization, phase-diversity camera, or explicit object optimization. Those are distinct experiments, not unimplemented runtime stages.

The local implementation preserves the selected source's numerical conventions, including its 54-slot basis with one identically zero slot, entire zero-frequency-axis exclusions in the reduced objective, scalar channel noise variance during wavefront inference, per-frame weighting during final image estimation, and unchanged Adam learning rate. These details must not be replaced with superficially similar textbook formulas.

## 2. Symbols, units, frames and regions

Let `k=0,1` identify broadband and narrowband objects; `j=0,...,11` identify simultaneous exposure pairs; `p=0,...,120` identify isoplanatic regions in row-major order; `m=0,...,53` identify coefficient slots. The detector array uses `(row,column)` order, unchanged from the supplied observational HDF5. `N=88` is the local image/pupil side length. The physical angular sampling is `Delta=0.059 arcsec/pixel`, effective wavelength is `lambda=8542 Angstrom`, and clear-aperture diameter is `D=100 cm`. Both passbands use the same effective wavelength in this selected model.

Raw calibrated camera intensities `Y[k,j,r,c]` are float32 in upstream digital intensity units. No absolute radiometric scale is supplied. Normalize every complete 512-square frame before extracting regions:

```math
\mu_{kj}=512^{-2}\sum_{r,c}Y_{kj}(r,c),\qquad
d_{kj}(r,c)=Y_{kj}(r,c)/\mu_{kj}.
```

The means and divisions are NumPy float32 operations. Region `p=11u+w` begins at detector `(40u,40w)` for `u,w=0,...,10` and contains `[40u:40u+88,40w:40w+88]`. There is no extension or padding. The 24 trailing rows/columns outside the union `[0:488,0:488]` still affect full-frame means. Every one of the twelve frames and both channels participates in inversion.

Extraction uses the same unfolded memory ordering as upstream: for one channel the `(121,12,88,88)` tensor has strides `(1,937024,10648,121)`. Keeping this convention matters because float32 covariance multiplication followed by a near-singular-value noise estimate can change with BLAS layout, despite equal array values. It is an explicitly validated numerical detail, not a new acquisition assumption.

## 3. Apodization and estimated noise

The local circular-convolution model would otherwise join unrelated opposite image edges. Construct a length-88 window equal to one except at each six-pixel border, where it follows the respective half of a length-12 NumPy Hann window. Its outer product is `W[r,c]`. For each local frame with spatial mean `bar d`, use

```math
\widetilde d_{kpj}=(d_{kpj}-\overline d_{kpj})W+\overline d_{kpj}.
```

No linear gradient is removed. The reference does not claim exact conservation of an arbitrary nonconstant frame's mean under tapering; it preserves a constant field and reduces boundary discontinuities. `cosine_window` constructs the float32 window, then the solver moves it and patches to the selected reconstruction device. `apodize` computes Torch means and arithmetic there, following the original window-device convention. The prepared frames are copied to a contiguous CPU buffer for the orthonormal FFT, and the resulting spectra are moved to the reconstruction device. This GPU-taper/CPU-FFT split is intentional: CPU and GPU mean accumulation differ slightly and affect nonconvex continuation.

Noise is estimated on **untapered** normalized local frames. For a frame matrix `X` of size `M by L`, here `M=L=88`, compute the spatial covariance `C=XX^T` and singular values `s_l` of `C`. Put `beta=M/L`,

```math
q=\frac{8\beta}{\beta+1+\sqrt{\beta^2+14\beta+1}},\qquad
\ell=\sqrt{2(\beta+1)+q},\qquad
\omega=0.56\beta^3-0.95\beta^2+1.82\beta+1.43,
\qquad
\sigma_{kpj}=\frac{\mathrm{median}_l\sqrt{s_l}}{\sqrt L\ell/\omega}.
```

This is the selected singular-value noise estimator. Its assumptions approximate low-rank solar signal plus additive Gaussian noise; spatial signal texture, detector correlations and float32 conditioning can bias it. It is not independently measured instrument noise. NumPy batched SVD is a permitted linear algebra primitive; the covariance and conversion to noise are local mathematics.

## 4. Explicit pupil and phase basis

The dimensionless pupil overfill is

```math
g=206265\frac{\lambda\,10^{-8}}{D\Delta},
```

where wavelength is in Angstrom and diameter in cm. Preserve `lambda=np.float32(8542.0)` and NumPy's resulting scalar conventions. The aperture uses integer pupil coordinates `x,y=-44,...,43`:

```math
A(x,y)=\mathbf 1\left[\sqrt{x^2+y^2}/44\le 1/g\right].
```

There is no central obstruction or spider. The phase-basis coordinates intentionally differ: use 88 inclusive samples from -1 to 1 along each axis, radial coordinate `rho=g sqrt(x^2+y^2)`, angle `theta=atan2(y,x)`, and support `rho<=1`. This is the discretization shipped by upstream, rather than silently replacing both grids with a common half-pixel convention.

For one-based Noll index `jz`, derive radial order `n` and signed azimuthal order `s` by the Noll ordering in `noll_indices`. With `m=abs(s)` and `q=(n-m)/2`, evaluate the radial Zernike polynomial explicitly through a general Jacobi polynomial:

```math
R_n^m(\rho)=(-1)^q\rho^m P_q^{(m,0)}(1-2\rho^2).
```

For `s=0`, `Z=sqrt(n+1)R`; for `s>0`, `Z=sqrt(2)sqrt(n+1)R cos(m theta)`; for `s<0`, use the corresponding sine. The public theoretical mixing table contains records `(ell,jz,c)` and is independent of the solar target. Compute

```math
B_m(\rho,\theta)=\mathbf 1[\rho\le1]
\sum_{(\ell,jz,c):\ell=m+2}cZ_{jz}(\rho,\theta),\qquad m=0,\ldots,52.
```

Set `B[53]=0`. This preserves an upstream off-by-one in its allocation/loop and is explicitly treated as a null gauge rather than a physical 54th independent atmospheric degree of freedom. The effective phase space has 53 nonzero basis functions. It is scientifically misleading to claim recovery of the null coefficient, so the output contract fixes it to zero.

The pupil phase is `phi[p,j]=sum_m a[p,j,m]B[m]` in radians. There is no piston. Remove the translation gauge before every forward evaluation by subtracting the first exposure's two tip/tilt coefficients from those of every exposure in the same region. All higher coefficients are left unchanged. Final output uses this canonical form; the uncentered optimization variable itself need not have zero first-exposure tip/tilt.

## 5. PSFs, optical transfer functions and transform normalization

Write `F_o` for the orthonormal two-dimensional Torch FFT. It uses the negative-exponential forward sign and divides the ordinary FFT by `N`. For a generalized pupil `P=A exp(i phi)`, compute

```math
E=\mathcal F_o(P),\quad
h=|E|^2/\sum_r|E(r)|^2,\quad
S=\mathcal F_o(h).
```

The PSF is real, nonnegative, unit flux and stored with its origin at array index `(0,0)`; no FFT shift is introduced for convolution. `S[0,0]=1/N`, not one. The observed local-frame Fourier data are `I=F_o(tilde d)`. The final object spectrum uses the **ordinary** forward transform, so the observation equation `I=O S` is dimensionally and numerically consistent. The final inverse is the ordinary inverse FFT, which divides by `N^2`. Multiplying an orthonormal OTF by an ordinary object spectrum is intentional; changing only one normalization changes reconstructed intensity scale.

The radial image-frequency coordinate is `rho_f=sqrt(fx^2+fy^2)/f_cut`, with `f_cut=D/(lambda*1e-8)/206265` in cycles/arcsec and `fx,fy=np.fft.fftfreq(N,d=Delta)`. Construct a diffraction mask `M=1` below 0.85, zero above 0.90, and `cos[(pi/2)(rho_f-0.85)/0.05]` in between. This regularizes unsupported spatial frequencies without assuming a sharp solar image.

## 6. Reduced wavefront objective and its normalization

The selected joint objective uses constant object spectral power `U=100`. For each channel, use one scalar `v_k=mean_{p,j}(sigma[k,p,j]^2)` over the complete set being optimized. Let

```math
Q_{kp}=v_k+U\sum_j |S_{pj}|^2,\quad
D_{kp}=\sum_j|I_{kpj}|^2,\quad
C_{kp}=\sum_j I_{kpj}\overline{S_{pj}}.
```

The wavefront objective is

```math
L(a)=\sum_k\frac{1}{121(87)^2}
\sum_p\sum_{u=1}^{87}\sum_{v=1}^{87}
M(u,v)\frac{D_{kp}(u,v)-U|C_{kp}(u,v)|^2/Q_{kp}(u,v)}{2v_k}.
```

The frequency indices are unshifted. The `1:` slices exclude both complete zero-frequency axes. This differs from excluding only one DC bin; it is a disclosed selected-code convention, independently tested rather than silently explained as DC removal. The mathematical expression follows analytic elimination of an image with a Gaussian spectral prior. The reference's final image calculation uses additional per-frame weighting and support filtering described next, so it must not be asserted to be exactly the minimizer of this scalar-variance loss.

There is no additional wavefront or object regularizer in this selected configuration: the configured IUWT weight is zero, no IUWT operation is called, and positivity enforcement is disabled. The object prior, diffraction support and complexity continuation remain active regularizing mechanisms. Automatic differentiation computes derivatives through the locally implemented pupil, normalization, FFT and reduced objective; it does not call a domain reconstruction library.

## 7. Initialization, activation order and Adam updates

Initialize all `121*12*54` coefficient slots to zero. Begin with two active tip/tilt modes. The radial-degree endpoints are `[2,5,9,14,20,27,35,44,54]`. For iteration index `t=0,...,249`, the planned schedule is two modes while `t<25`, 54 while `t>150`, and otherwise endpoint `floor(8(t-25)/125)` with clipping to `[0,8]`. **Apply this scheduled active count only after the current optimization step**, exactly as the original code does. The first loss uses two modes; coefficients of inactive modes have no loss dependence.

For gradients `g_t`, the permitted general Adam optimizer implements

```math
m_t=0.9m_{t-1}+0.1g_t,\qquad
v_t=0.999v_{t-1}+0.001g_t^2,
\qquad
a_{t+1}=a_t-0.08\frac{m_t/(1-0.9^{t+1})}
{\sqrt{v_t/(1-0.999^{t+1})}+10^{-8}}.
```

Torch Adam uses its default beta, epsilon, zero weight decay and ordinary non-AMSGrad mode. Every iteration clears gradients, evaluates centered active coefficients, computes both channel OTFs and the shared loss, backpropagates, and steps the optimizer. Stop after exactly 250 updates; there is no line search, early stopping or convergence-tolerance trigger. A cosine learning-rate scheduler is instantiated upstream but never stepped, so its learning rate remains 0.08; the local code omits that inert object. The zero initial modes and deterministic input require no hidden seed. The CLI fixes Torch CPU threads to one; CPU-only execution is supported, while full benchmark runs are validated on the documented GPU.

## 8. Analytic final objects and spectral support

After the last step, evaluate all 54 coefficient slots with the canonical gauge. For each channel and region put `v[k,p]=mean_j sigma[k,p,j]^2` and dimensionless frame weights

```math
\gamma_{kpj}=v_{kp}/(\sigma_{kpj}^2+10^{-10}),\quad
T_{kp}=\sum_j\gamma_{kpj}\overline S_{pj}S_{pj},\quad
G_{kp}=\sum_j\gamma_{kpj}\overline S_{pj}I_{kpj},\quad
O_{kp}=G_{kp}/(T_{kp}+10^{-10}).
```

The epsilon is part of the preserved convention. Form `H=real[T/(conj(G)G)]`, FFT-shift its spatial-frequency axes, and apply a 3 by 3 median with periodic wrap. Compute `f=1-v[k]H` using the **global channel mean noise variance**, not the region-specific variance used in the frame weights; set values below 0.2 to zero, and clip values above one to one. From the central frequency seed, retain the eight-connected component admitted by the original flood tolerance 0.9 applied to `1-f`. This is a **binary support**, not the real-valued `f` itself. Shift back and multiply by diffraction mask `M`. The local `scipy.ndimage.label` computes connected components explicitly; it is not a reconstruction shortcut.

The final local intensity is `o=real(ifft2(O*support*M))` using the ordinary inverse FFT. Negative values are retained if produced; positivity ADMM was disabled upstream. This filtering order differs slightly from a prose description that smooths an already clipped filter. The code's order is preserved and validated with independent numeric fixtures.

## 9. Full-scene mosaicking and declared outputs

Discard six pixels from each local edge, leaving 76-square interiors. Use the outer product of a 76-element window, with each 30-pixel edge following half of a 60-element Hann window. Place interiors at `(40u,40w)`, sum weighted values and divide by the weight sum plus `1e-10`. The output is 476-square, covers detector indices `[6:482]` in each direction and has a zero outermost row/column because the Hann endpoints vanish. The small denominator epsilon attenuates extremely low-weight edge pixels; no ad hoc filling is performed.

Return both channel mosaics, all local object arrays and all canonical coefficient arrays in one batch-first NPZ. `loss_history` and runtime are maintainer results returned by the Python solver but are not submitted arrays. No calibrated truth is available. A separately reproduced upstream result is the hidden algorithm reference, identified as such; conventional MOMFBD products are comparison measurements, not truth.

## 10. Verification, evaluation and limitations

Stage parity freezes upstream normalization/extraction, noise, taper, phase basis, PSF/OTF, reduced loss, autograd, one Adam update and projected objects before accepting local stages. Independent tests cover known Noll orders, nonnegative unit-flux PSFs, OTF DC scale, circular-operator adjoint, double-precision loss directional derivatives, gauge invariance, null-slot invariance, activation schedule and mosaic coordinates. The local float64 loss accumulator supports derivative checks without changing float32 benchmark behavior.

Full fresh reconstructions must pass an independent Judge against the upstream result and preserve scene-wide detail. The Judge measures every channel's intensity arrays, pupil-coefficient-induced PSF agreement, translation/null gauges, and exact declared mosaic consistency. The physical phase can have intensity-only ambiguities, so raw coefficient Euclidean distance is not called physical accuracy. Reference parity, even perfect parity, cannot establish the Sun's true unblurred intensity or unique atmospheric phase.

The selected burst has only twelve exposure pairs, finite phase bandwidth and approximate local shift invariance. Patchwise regularization can create mosaic structure at very high spatial frequencies; spectral power should therefore be interpreted cautiously. Unknown detector correlations, residual fixed-pattern calibration, scan evolution, model-vs-observation passband approximations and the discrete aperture/basis mismatch limit physical interpretation. These limitations remain part of a substantial joint nonlinear inverse problem rather than grounds for replacing it with a smaller demonstration.

The diffraction-mask cosine transition is evaluated on the selected reconstruction device, as upstream. CPU and CUDA cosine differ by up to one float32 ULP at 92 transition pixels on this grid. Generating that mask on CPU and then transferring it is not numerically equivalent for the continued nonconvex benchmark; GPU stage fixtures exercise the selected-device convention.
