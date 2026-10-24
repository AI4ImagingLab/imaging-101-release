# Implemented canonical homodyne reconstruction

## Physical scale, observations and calibration

The benchmark reuses the333-heralded shots in all100 original files sorted by integer timestamp. There are3,756,232 raw shots, not the cached state matrices in the author release. The public NPZ is a lossless, non-object conversion of existing measured arrays. The three photon-number outcomes and their confidence distances identify the conditional ensemble; no synthetic state generator exists.

The canonical convention is hbar1, [Q,P]=i, and vacuum variance1/2. The older2004 method paper uses [x,p]=i/2, so its quadrature coordinate corresponds to native q/sqrt(2). This change of canonical coordinates explains the different Gaussian/Hermite normalization; the local runtime preserves the2025 author's actual convention throughout.

For each acquisition block f, let mu be its measured homodyne mean and V the64 independent vacuum variances. A raw ADC sample s becomes q=(s−mu)/sqrt(2 mean(V)). All arithmetic is float64 and retains original per-file ordering. A raw phase index multiplies that file's actual stored phase increment. There is no sample clipping, phase-window rejection or Wigner-window rejection.

Fit each block's unconditional signal variance divided by its phase-resolved vacuum variance and multiplied by1/2. With raw phase x, phase offset b, distortion amplitude a, distortion phase c and extrema u,v, the model is

```math
\vartheta(x)=x-b+a\sin(x-c),\qquad
V(x)=u\cos^2\vartheta(x)+v\sin^2\vartheta(x).
```

Use the original unbounded `scipy.optimize.curve_fit` with its default least-squares settings. The initial b is the minimum-variance phase shifted to[-pi/2,pi/2); initial a=c=0, u=min(normalized variance), v=max(normalized variance). No conditional reconstructed state enters this calibration. All100 fitted five-parameter vectors are tested against the original capture.

The native confidence rule creates201 equally spaced histogram bins independently for each detector's float64-cast distances. It selects the bin centre whose cumulative histogram fraction is closest to0.8, preserving NumPy's first-minimum convention. Keep shots strictly below all three centres. The resulting centres are the public fixed cutoffs, yielding1,881,028 accepted shots. This rule defines the effective conditional state; it is not an inverse regularization parameter. Local selection indices and calibrated samples are independently byte-equal to the original.

## Born model and positive list-mode inverse

Let n=0,...,30 denote photon number. At shot k with canonical quadrature q and calibrated phase theta, the column A has elements

```math
A_n=\frac{H_n(q)e^{-q^2/2}e^{in\theta}}{\pi^{1/4}\sqrt{2^n n!}},\qquad
p_k=\mathbf A_k^\dagger\rho\mathbf A_k.
```

H is the physicists' Hermite polynomial. The implementation evaluates normalized amplitudes with the original recurrence: first Gaussian ground state, then sqrt(2)q exp(i theta) times that state, and then adjacent photon levels with the sqrt(2/n) and sqrt((n−1)/n) factors. It stores all31×N complex128 amplitudes. No histogram approximation or shot subsampling enters the selected inverse.

The objective is the unnormalized log likelihood sum(log(p)), without a regularizer or additive numerical floor. A full-rank identity/31 initializes rho. Each iteration computes

```math
R=\sum_k\frac{\mathbf A_k\mathbf A_k^\dagger}{p_k},\quad
R\leftarrow(R+R^\dagger)/2,\quad
C=R\rho R,\quad C\leftarrow(C+C^\dagger)/2,\quad
\rho^{\mathrm{new}}=C/\mathrm{Tr}(C).
```

The R scaling by sample count is omitted exactly as in source; it cancels under trace normalization. The old Born probabilities supply that iteration's recorded likelihood, while the returned density has received its update. Both Hermitian symmetrizations and the trace division are retained in the same multiplication order. The method uses exactly100 iterations, with no convergence early exit, line search, generic optimizer, PSD projection, detector-loss inversion or fitted phase-space alignment. Finite photon support and finite iteration budget are the selected numerical constraints.

The source's step diagnostic takes a matrix square root S=sqrt(rho_new), divides S by trace(S@S), and evaluates trace(sqrt(S@rho_old@S)) squared. This slightly redundant normalization for unit-trace states is retained, rather than replaced by a differently normalized fidelity. The diagnostics are real-if-close using the source's absolute tolerance convention. The full measured run empirically increases likelihood each step; no universal monotonicity guarantee is claimed for every possible R-rho-R dataset.

Snapshots after1,2,7,25,50,100 updates retain actual computed matrices. They are not separate reconstructions or hidden checkpoints used to initialize the inverse.

## Wigner field and additional observables

The Wigner array uses rows P, columns Q; each coordinate is linspace(-7,7,201). The full-plane transform integrates to one, but its finite-window numerical integral need not be exactly one. The output is never rescaled by its finite-window sum.

The local field evaluates the original coupled oscillator-basis recurrence. With radius r²=Q²+P², Gaussian g=exp(-r²)/pi and complex beta=Q−iP, the equivalent basis expansion is

```math
W=\sum_n\rho_{nn}(-1)^n L_n(2r^2)g+
2\operatorname{Re}\sum_{d=1}^{30}\sum_{n=0}^{30-d}
\rho_{n+d,n}(-1)^n\sqrt{\frac{n!}{(n+d)!}}
(\sqrt2\,\beta)^d L_n^{(d)}(2r^2)g.
```

Here L denotes ordinary or generalized Laguerre polynomials, and d is an off-diagonal photon separation. The source/local runtime uses its stable recurrence; the independent Judge evaluates this closed Laguerre expression using general SciPy special functions, giving a separate transform-consistency check. Analytic vacuum, one-photon and complex coherent states independently verify normalization, negative origin and the sign of momentum displacement. Number-state Hermite norms are integrated on[-12,12] for validation only; the benchmark output window remains unchanged.

The quadrature visualization helper explicitly retains the author's finite-grid normalized probability masses. Converting them to a plotted density requires division by grid spacing and must be labeled as finite-window normalization. They are not substituted for the continuous Born density in the likelihood.

Displacement observables use the closed matrix element exp(-|alpha|²/2) sqrt(m!/n!) alpha^(n−m) L_m^(n−m)(|alpha|²) for n>=m, with the conjugate/sign counterpart for n<m. The alpha0 case is the identity by its mathematical limit. This is a direct readable formula using general special functions; no quantum library is imported. Native stabilizer displacements are alpha=1.86/sqrt(2) and alpha=i sqrt(2)pi/1.86. Historical cached stabilizer reports remain labeled separately from fresh values.

## Comparisons, evaluation and scientific limits

The standard answer is the complete calibrated31-state/100-update inverse. Notebook-only lower photon supports and uncorrected-phase reconstructions are explicit fresh sensitivity experiments, using the same measured shots and the stated changed approximation. Lower-dimensional matrices can be padded with exact zero unsupported levels for a common comparison; this is a representation convention, not an absent method filled with zeros. Historical release matrices for111,133,335,444 and333 are context only; their acquisition/training details are not independently certified. They never replace the measured benchmark data or its fresh result.

There is no actual experimental state truth. An independently executed unchanged original run supplies the method reference. The density metric is complex Frobenius difference divided by reference Frobenius norm; the Wigner metric is RMS difference divided by reference dynamic range. Every matrix/field entry is evaluated without alignment. Frozen bounds are1e−6 and1e−7 respectively, exceeding actual valid complex64/float32 serialization errors2.2112e−8 and3.97135e−9 by at least ten. These are nonzero numerical tolerances, not self-comparison-zero quality limits. Hermitian relative error and trace error must be<=1e−6, minimum Hermitian eigenvalue>=−1e−6, and independently computed rho-to-Wigner maximum error<=1e−6. Separate source parity remains the predeclared2e−8 scaled full-field bound.

Effective-state/model-reference agreement is not certified preparation fidelity. Finite photon support, finite window, phase-calibration model, postselection, finite samples and uncorrected detection losses limit interpretation. The public raw and calibration arrays suffice for reconstruction; neither hidden matrices nor historical release values are solver inputs.
