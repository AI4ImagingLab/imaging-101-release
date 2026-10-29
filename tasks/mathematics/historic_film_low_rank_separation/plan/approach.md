# Implemented principal component pursuit

The complete native film has T=84 digital frames, H=240 rows and W=320 columns. Let I(t,y,x) be uint8 grayscale intensity. Define the float64 matrix D[yW+x,t]=I(t,y,x)/255, shape (76800,84). All components are in normalized digital intensity units. There is no centering, mean subtraction, registration, border masking, spatial/temporal interpolation or simulated corruption. Physical pixel size and original exposure rate are unknown. Container playback29.97002997Hz is not an original acquisition clock.

## Objective and explicit updates

Solve min ||L||_*+lambda||S||_1 subject to D=L+S. Nuclear norm is sum of singular values; entrywise L1 norm is sum of absolute entries. No per-pixel objective normalization is applied. The augmented Lagrangian is ||L||_*+lambda||S||_1+<Y,D-L-S>+(mu/2)||D-L-S||_F², with real dual matrix Y. Thus the dual sign and both positive Y/mu shifts below are fixed.

Start S=0,Y=0. Set mu=D.size/(4*sum(abs(D)))=0.9481384165894786 and lambda=1/sqrt(max(D.shape))=0.0036084391824351605. For each iteration:

```math
L^{k+1}=\operatorname{SVT}_{1/\mu}(D-S^k+Y^k/\mu),
\quad S^{k+1}=\operatorname{soft}_{\lambda/\mu}(D-L^{k+1}+Y^k/\mu),
\quad Y^{k+1}=Y^k+\mu(D-L^{k+1}-S^{k+1}).
```

For real z, soft_tau(z)=sign(z)max(abs(z)-tau,0). If X=U diag(sigma)V^T, SVT_tau(X)=U diag(max(sigma-tau,0))V^T. Both proximal maps are implemented locally; mu is fixed, with no continuation, generic inverse solver or changed regularizer. Initialization, signs and update ordering match the selected original implementation. There is no line search.

`singular_threshold` computes the identical proximal formula through a small Gram eigendecomposition: for tall X, diagonalize X^T X=V diag(sigma²)V^T, retain only sigma>tau, and return (X V) diag(1-tau/sigma)V^T. The wide case uses XX^T analogously. This avoids forming unused tall singular vectors; it is not a rank cap, randomized SVD or different inverse method. Roundoff below the strictly positive shrinkage threshold cannot enter the active solution. Independent proximal optimality/equivariance cases, frozen first-step source SVD outputs and every full final component and residual trace validate the numerical implementation. All math is float64.

Stop when ||D-L-S||_F<=1e-7||D||_F, evaluated after the dual update, or after3000 iterations. Record both convergence boolean and relative residual; never call a maximum-iteration stop converged. The original default1000 limit failed the same tolerance at2.166287e-7. Before local extraction, the original budget was increased to3000 without changing tolerance or method: it converged in1657 iterations at9.999964969747064e-8. The canonical local result matches the full source run; independent source repeat establishes numerical reproducibility. No historical paper accuracy is claimed for this film.

## Primary preprint versus selected implementation

The actual rendered arXiv0912.3599v1 PDF page29 was independently reviewed, not merely its extracted text. Its Eq5.1 prints the positive dual term and mu/2 quadratic, but Eq5.2/5.3 and Algorithm1 print thresholds lambda*mu and mu, and the L step has a negative Y/mu shift. These printed steps are inconsistent with that same stated objective. Completing the square for fixed S gives (mu/2)||L-(D-S+Y/mu)||_F² plus the nuclear norm and constants: the correct shift is positive and threshold1/mu. The S subproblem analogously has threshold lambda/mu. Independent nonzero-dual/mu=2 objective tests distinguish these conventions. The selected original code already implements these internally consistent updates. Local extraction preserves that original executable method, rather than silently copying the preprint inconsistency, changing the source algorithm, or claiming character-for-character agreement with the printed algorithm. The published JACM DOI identifies the method lineage; this discrepancy is specifically established for the archived v1 preprint, not asserted for an uninspected final publisher version.

## Measurements, comparisons and limits

Both signed full components are required. The evaluator uses each reference-relative Euclidean error with1e-6 upper boundary and the exact visible-data additive residual with1.1e-7 upper boundary, fixed before local extraction. No mean subtraction, dynamic-range normalization, magnitude conversion or hidden spatial/intensity alignment occurs. Reference is a standard-method reconstruction, never truth. Tight nonzero reference tolerances reflect independent numerical reruns rather than mechanically multiplying a zero self-error.

The notebook actively reconstructs the entire84-frame input using the same defaults. Its further actual reconstructions multiply lambda by0.75 and1.25 on the entire movie, and retain every2nd/4th digital frame without spatial changes. Every variant records iterations, actual convergence status, rank at relative singular-value threshold1e-6, sparse fraction at absolute normalized intensity1e-3, both energies and additive residual. The thresholded rank/sparse fraction are descriptive statistics, not inferred truth. The fresh canonical result supplies the lambda1 point. Fresh full-data rank1/3/5 truncated PCA residuals are explicitly alternative decompositions. Temporal experiments alter digital sampling, not exposure or verified historical camera rate. Nonconverged variants remain labeled in figures and machine results.

Four figures preserve a common intensity scale across raw/native components, signed sparse maps centered at zero, source/local convergence, all-frame energy/residual curves, singular spectra, full-data lambda sensitivities, PCA comparisons and temporal sampling. Selected frames are merely display panels; all84 are reconstructed and evaluated. Persistent moving people, jitter and film artifacts can enter either component. There is no spatial foreground truth, segmentation accuracy, restored-film quality metric or calibrated uncertainty interval.
