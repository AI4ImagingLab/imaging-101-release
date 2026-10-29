# Implemented conditional absorption tomography

The binding is all 64,332 real CLAMATO DR1 v4 continuum-normalized pixels, not a newly generated mock or cropped demonstration. Their 240 background sources sample a continuous foreground transmitted-flux fluctuation field. The output is the native unsmoothed 60×48×876 cell-centered Wiener field, dimensionless, in increasing comoving x/right ascension, y/declination, z/line-of-sight order. Coordinates use Mpc/h. No periodic extension or clipping is applied; slightly exterior sightlines remain valid. There is no independent field truth.

## Observations, coordinates and covariance

Each row has position r, released standard deviation sigma and contrast d=f/(C Fmean)−1. C and Fmean are author preprocessing, not estimated in this task. The author already imposed sigma≥0.2. The measurement model is d=s(r)+epsilon with independent zero-mean noise covariance N=diag(sigma²). Continuum uncertainty and correlated instrument systematics are not fully represented by this simplified noise model.

Grid coordinates are `(i+1/2)Lx/nx`, `(j+1/2)Ly/ny`, `(k+1/2)Lz/nz`, with L=(30,24,438) Mpc/h and n=(60,48,876), yielding 0.5 Mpc/h centers and z-fastest flattening. The fixed cosmology conversion is already present in input positions. The README specifies the origin, cosmology and fixed derivative; reconstruction does not apply another coordinate transformation.

For positions a,b, define q=((ax−bx)²+(ay−by)²)/(2Lperp²)+(az−bz)²/(2Lparallel²), with Lperp=2.5 Mpc/h and Lparallel=2.0 Mpc/h. K(a,b)=v g(q), v=0.05. The source uses a table, not a direct Gaussian: h=8/199, tj=exp(−jh), u=q/h, j=floor(u); g(q)=tj(j+1−u)+t(j+1)(u−j) for j≤198 and g(q)=0 otherwise. Thus q≥8 is truncated, including q=8. Relative interpolation error and the small finite cutoff discontinuity are separately tested against the ideal Gaussian. No assertion of an exactly Gaussian continuous covariance is made for this discretization.

The unknown conditional map is m=K(G,D)(K(D,D)+N)⁻¹d. G denotes grid points and D observation positions. In the positive-definite ideal covariance model, this is also the Gaussian conditional mean and the minimum of the corresponding quadratic prior/data objective. The finite table approximation is preserved from source; independent noisy-system eigenvalue and symmetry checks accompany parity rather than assuming the tabulated kernel is a universal positive-definite kernel.

## Whitening and explicit inverse updates

Let W=diag(w), w=1/sigma; b=Wd; A=I+W K(D,D) W. Solve A y=b, then alpha=W y and m=K(G,D) alpha. The quadratic solve objective is yᵀAy/2−bᵀy; its gradient is Ay−b. The positive diagonal preconditioner P has entries 1+v w².

Zero initialization: x=0, r=b−Ax, z=P⁻¹r, p=z, delta=rᵀz. Here x is the current whitened coefficient iterate; it is renamed y after solving. Each custom PCG step computes qvec=Ap, a=delta/(pᵀqvec), x←x+a p. Except at every 50th update, r←r−a qvec. At those refresh updates, r←b−Ax. Then z←P⁻¹r, delta_new=rᵀz, beta=delta_new/delta, p←z+beta p and delta←delta_new. The qvec vector is distinct from the scalar covariance separation q.

The shipped defaults are max_iter=500, tol=0.001 and step_r=50. The source loop starts at 1 and uses iteration<max_iter, hence at most 499 updates. Stop when the Euclidean residual norm is ≤tol times the Euclidean norm of b. The full native original stops at update 22, before the refresh branch occurs. A separate upstream branch diagnostic explicitly forces a refresh at update 3 on a nonconverged numerical case; this is not the unchanged default physical pipeline. The earlier experiment forcing 50 steps on an already converged five-step small case is archived privately as an invalid parity-case design, not an accepted result. Its roundoff-amplified search directions are excluded from meaningful stage comparison. No stage tolerance was relaxed.

The zero-right-hand-side case returns the zero field with zero iterations and success=False, matching the source's early-return numerical field. The source leaves its iteration-count field uninitialized in this branch; the cleaned diagnostic explicitly defines it as zero. This metadata correction does not change real-data mathematics. No line search or proximal update is part of PCG. Tighter-tolerance source solves quantify stopping-error sensitivity separately from the default result.

## Sparse evaluation without a domain reconstruction dependency

`cKDTree` is used only to locate pairs with scaled distance ≤4, where scaling is (Lperp,Lperp,Lparallel), equivalent to q≤8. Physical coordinate differences are then recomputed to evaluate the explicit kernel. All retained entries are assembled as a sorted CSR matrix. The sparse product applies A to a vector as vector+w*(K@(w*vector)); the whitened matrix itself is not formed. This preserves the source's finite-support operator while avoiding its exhaustive zero-valued pair loops. Generic spatial indexing and sparse arithmetic do not perform an inverse solve.

After solving, observation predictions K alpha are saved as diagnostics. The large observation covariance is released before map evaluation. Grid-to-data covariance is built in chunks of 1,024 grid points and multiplied by alpha; all grid points are evaluated. The output is reshaped with z fastest. Chunking is a memory control and does not alter the model, masks or grid. Full, signed residuals and raw versus additional-smoothed maps must be labeled separately in the notebook.

## Scientific evaluation conventions

The hidden reference is the independently executed standard method, never true cosmic density or flux. Numerical limits were frozen from independent original repeat, float32 serialization and tighter original-PCG stopping sensitivity before scoring the full cleaned output. The tighter original solve differs by NRMSE0.00014437187338231305 and NCC0.9999988674200637. The fixed limits are NRMSE≤0.00021655781007346957 and NCC≥0.9999983011300956, including1.5 times measured stopping error and a measured serialization floor. NCC is flattened cosine similarity without mean subtraction; NRMSE is RMS difference divided by the reference maximum minus minimum. The complete unaligned real field is evaluated. Zero, uniform, shuffled and strongly oversmoothed fields fail these fixed limits. No foreground galaxy positions are truth labels.

The released nominal noise floor0.2 appears as0.20000000298023224 in double storage, reflecting the author's original single-precision floor. Values are preserved exactly rather than rounded/replaced by a double literal.

The paper's reconstructed-map variance K(G,D)(K(D,D)+N)⁻¹K(D,G) is not the posterior covariance K(G,G) minus that expression. The selected native configuration disables variance calculation. Consequently any notebook coverage proxy or sensitivity spread must be explicitly labeled as such, rather than calling it that paper variance or physical uncertainty. Historical simulation signal-to-noise and true-field comparisons from the paper cannot be measured on the supplied real-only observations and must remain labeled historical context.
