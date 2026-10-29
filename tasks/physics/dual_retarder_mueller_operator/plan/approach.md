# Implemented full-operator inverse

The bound task is the unchanged author-native FullMuellerExample's complete24-angle acquisition. The original program generates a physical retarder-times-polarizer sample, propagates unit unpolarized input through known dual-rotating quarter-wave optics, and recovers its full4×4Mueller matrix. The unknown is nevertheless all16matrix entries: the cleaned inverse does not assume the sample's generating family. The source's16-angle acquisition is retained as a separate rank-deficient aliasing comparison, not silently substituted for the benchmark.

## Conventions and local acquisition

All optical angles/phase delays are radians. Stokes column vectors have basisI,Q,U,V, with circular parameter $V=2\mathrm{Im}(E_xE_y^*)$. The sample map is $s_{out}=Ms_{in}$. Its entries are real signed dimensionless intensity-transfer coefficients. Ideal polarizers preserve their physical factor1/2; no matrix normalization is applied. Known incident Stokes intensity before the generator is1, detector gain1and offset0. The native author case adds no measurement noise. All24endpoint-inclusive angles span0toπ; analyzer retarders rotate at5times the generating angle, with both phase delaysπ/2and both polarizers at0.

`linear_polarizer` and `linear_retarder` evaluate the explicit trigonometric matrices in README. For angle θ define generator $G=R(\theta,\pi/2)P(0)$ and analyzer $A=P(0)R(5\theta,\pi/2)$. The rightmost optic acts first. Let $g=Ge_0$ and $a^T=e_0^TA$, with $e_0=(1,0,0,0)^T$. The measured power is $p=a^TMg$.

For row-major vector $m=\mathrm{vec}_{row}(M)$ the design row is $W_{n,4i+j}=a_{n,i}g_{n,j}$, where i,jindex output/input Stokes components0to3. Thus the entire acquisition is $p=Wm$, with Wshape24×16. Explicit local outer products reproduce original Katsu broadcast-Kronecker rows. The adjoint returns the4×4reshape of $W^Ty$. Independently derived Jones-to-Mueller tests verify signs and physical factor1/2, beyond agreement of two Mueller implementations.

## Selected inverse and numerical conventions

The objective is unweighted $\frac12\|Wm-p\|_2^2$, with no regularizer, physical projection, sample-family constraint, training, iterative update or line search. Compute the reduced SVD $W=U\Sigma V^T$. Keep singular value s only when $s>10^{-15}s_{max}$, matching the native NumPy pinv default, and evaluate

```math
\hat m=V\Sigma^+U^Tp.
```

The local inverse explicitly forms reciprocal retained singular values and matrix products; no polarimetry/domain solver is imported. Closed-form inversion needs no artificial iteration. The original24-angle case has numerical rank16and condition18.1044397903862, recovering actual synthetic truth at approximately1e-15error. The original16-angle0to2πcase has rank12and condition approximately7.44e15, so its pseudoinverse returns a minimum-norm solution that fits its intensities but differs from true matrix by27.36%relativeL2. Preserve that limitation and comparison; do not improve it by changing the author's angle schedule or assuming sample structure.

## Synthetic data and true-reference provenance

The actual observation/truth fixture was captured from executing every code cell of the original author notebook before writing these local functions. A private seed fixes its otherwise unseeded three NumPy random draws, without changing the author's sample family, angle sets, calibration, generation equation or inverse. README/public data do not expose that seed, matrix, or target construction. The hidden true-reference matrix is the original program's simulated sample, not its reconstructed matrix and not a cleaned-code self-reference.

`src.generate_data.author_example` reproduces that authored setup locally for an independent generation/parity test. Its hidden docstring identifies all seed/object/noise details. It is not called by main or the active benchmark inverse, and never supplies a fixture used to validate itself. The original author's percent-by-element comparison divides by zeros in the true matrix and prints NaN/Inf; those warnings are preserved privately. Cleaned numerical reporting uses well-defined full-matrix NCC/NRMSE and absolute element error instead, explaining this metric correction without changing the inverse.

## Physics and diagnostic mathematics

Independent Jones matrices with the stated circular convention produce the same retarder/polarizer matrices. A lossless retarder preserves intensity and the polarized-vector norm, and an ideal polarizer is passive. Full recovered sample action is checked on deterministic physically polarized Stokes inputs: output intensity remains nonnegative, output polarized norm does not exceed intensity, and throughput is≤1for this native passive sample.

For a further operator diagnostic compute Hermitian coherency $H=\frac14\sum_{i,j}M_{ij}(\sigma_i\otimes\overline{\sigma_j})$. The σbasis is identity,Z,X,negative-Y, matching I,Q,U,V; overbar is complex conjugation. Identity Mueller gives coherency eigenvalues0,0,0,1and an ideal depolarizerdiag(1,0,0,0)givesI4/4. The physical native noiseless sample is positive semidefinite to roundoff. No physical projection replaces the published unconstrained inverse, including in noisy sensitivity experiments.

## Evaluation and experiments

Score every one of16entries, without a fitted scale, transpose, phase correction or alignment. NCC is ordinary flattened cosine similarity; NRMSE is RMS element error divided by truth dynamic range. A zero estimate receives NCC0. Add independently computed relativeL2forward-observation consistency using explicit calibrated Stokes vectors. The true matrix is nonzero/nonconstant, so reference scales are well-defined. Native noiseless error is roundoff, not a meaningful nonzero stochastic error; pre-extraction boundaries NCC≥.999999999,NRMSE≤1e-10and observation-relative error≤1e-10are nonzero conditioning-aware tolerances rather than1.1×zero baseline error.

Notebook comparisons actively reconstruct both native16/24cases, preserve all16matrix panels and both measured signal groups, compare rank/singular/null directions, and display true/original/fresh roles separately. Sampling subsets of the bound24scan are explicit ablations and never replace the complete benchmark. Calibrated-retardance perturbations and added intensity noise are newly computed sensitivity experiments, not experimental observations or instrument-confidence claims. Every plotted alternative is computed from local mathematics; no archived original reconstruction fills a missing fresh result.
