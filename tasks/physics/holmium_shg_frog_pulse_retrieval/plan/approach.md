# Implemented SHG-FROG retrieval

The unknown is the 512-sample temporal complex envelope of the real Ho:CALGO regenerative-amplifier output. The primary observation is the complete author-processed Regen trace, in delay/frequency order. No original detector images, historical reconstructed pulse or measured absolute energy are used in inversion. A separate attempted MPC binding is archived as a difficult failed stability candidate and is not mixed into this task.

## Discrete physics and units

Let $N=512$, frequency increment $h=407474458795.6243$ rad/s and $d=2\pi/(Nh)=3.0116847910804345\times10^{-14}$ s. Indices $j,k=0,\ldots,N-1$ define $t_k=(k-N/2)d$ and $\omega_j=(j-N/2)h$. The actual measured delays $\tau_m$ have increment 30.096 fs, rather than $d$. The signal frequency is $\Omega_j=2\omega_0+\omega_j$, with $\omega_0=903760270468365.0$ rad/s.

The physical Fourier pair implemented with phase-corrected SciPy FFT primitives is

```math
F(x)_j=\frac{d}{2\pi}\sum_k x_k e^{i\omega_jt_k},\qquad
B(a)_k=h\sum_j a_j e^{-it_k\omega_j}.
```

Thus $B(F(x))=x$ and $d\sum|x|^2=2\pi h\sum|F(x)|^2$. For spectrum $a$, define $H_{mj}=e^{i\tau_m\omega_j}$, undelayed envelope $E=B(a)$, delayed envelope $A_m=B(H_ma)$, signal $S_m=A_mE$ and predicted intensity $T_m=|F(S_m)|^2$. This is periodic trigonometric interpolation of fractional delays. No integer-delay approximation or interpolation of the primary data is performed. The forward model has a flat effective response and excludes explicit crystal phase matching; this is an approximation to the processed experiment, with residuals retained and reported.

All observations are relative dimensionless intensities. Before projection, the unconstrained field amplitude is absorbed by a positive detector gain. With measured intensity $D$, uniformly weighted least squares uses

```math
\mu=\frac{\sum DT}{\sum T^2},\quad
r=\sum(D-\mu T)^2,\quad
R=\sqrt{r/(MN\max(D)^2)},\qquad M=N=512.
```

`scaled_residual` evaluates these exact sums. The selected source weights are all one; no unknown noise variance is claimed. The gain is nonnegative because both intensity arrays are nonnegative. The measured maximum is one.

## Projection and complex gradient

For a signal $S$, compute $Q=F(S)$, replace its amplitude by $\sqrt{D/\mu}$ while preserving phase, assign phase zero when $|Q|=0$, and apply $B$. Denote the result $P$. For fixed $P$, the signal loss is $Z=\sum|P-S|^2$. With $\delta=P-S$, its real Euclidean complex gradient (twice the Wirtinger derivative with respect to the conjugate spectrum) for each delay is

```math
g_m=-\frac{4\pi h}{d}\left[H_m^*F(\delta_mE^*)+F(\delta_mA_m^*)\right].
```

A directional derivative is $\mathrm{Re}\langle g,v\rangle$, with the Hermitian inner product conjugating its first argument. The physical factors are tested both against frozen original stages and independent dense Fourier/directional finite differences. The arXiv v1 amplitude projection notation has inconsistent gain prefactors between the main text and supplement; the selected original code implements the physically verified constraint $|F(P)|^2=D/\mu$. No known incorrect prefactor is copied into the local mathematics.

## COPRA local and global stages

The standard run initializes an independent zero-phase Gaussian $E_k=\exp[-2\ln2(t_k/(50\,\mathrm{fs}))^2]$, then sets $a=F(E)$. It does not read any retrieved field. The original NumPy RandomState sequence has seed1234 for reproducible stage and full-pipeline parity. Independent seeds and widths are also evaluated; this seed is not the definition of physical correctness.

At initialization, the maximum squared signal-gradient norm over all delays is stored. A local sweep uses a random permutation of the 512 delays. For each row, evaluate the current signal, project with the gain held fixed for that sweep, evaluate $Z_m$ and $g_m$, update the current maximum gradient norm $q$, and set

```math
\gamma_m=Z_m/\max(q,q_{\mathrm{previous}}),\qquad a\leftarrow a-\gamma_m g_m.
```

The previous maximum is that of the preceding sweep. The first previous maximum is the initialization maximum. The sweep's trace array contains rows from different successive spectra. Its error is therefore an approximate selection statistic; the gain is updated from that mixed array at sweep end. It must not be plotted as the physical trace of one pulse.

The source switches permanently to global mode when its counter reaches **five** consecutive iterations without improving the stored best statistic. The paper's main text describes ten; the shipped implementation follows the selected source's five. Global mode evaluates the complete current trace, optimizes its gain and residual, and computes the signal-space residual gradient

```math
g_r=-\frac{4d}{2\pi h} B\left[\mu F(S)(D-\mu T)\right],\quad
\eta_r=\alpha r/\|g_r\|_2^2,\quad P=S-\eta_rg_r.
```

With $g_Z=\sum_mg_m$ evaluated for this fixed target and $Z=\|P-S\|_2^2$, update

```math
\eta_Z=\alpha Z/\|g_Z\|_2^2,\qquad a\leftarrow a-\eta_Zg_Z.
```

The standard $\alpha=0.25$ applies to both global steps. There is no line search, regularizer, thresholded support or additional smoothing. The run stops after1000 iterations. The code intentionally preserves the source's **old-input best-save** convention: a step's selection statistic is compared with the current best, then the input spectrum is stored when it improves, while the updated spectrum becomes the next iterate. For local sweeps, the statistic is approximate. A stored approximate best has its complete physical error recomputed at the end. In local mode, logging replaces the mixed-row approximation with the complete physical residual of the current field; the best-selection statistic remains the mixed-row approximation. In global mode the log retains the preceding input's complete error. Final scientific residual plots always recalculate the full trace of the returned spectrum.

## Comparison paths

PIE uses the same actual fractional delays and identical gain projection. A sweep draws one $\beta$ uniformly from0.1 to0.5, permutes the rows, and updates the temporal field by $E\leftarrow E+\beta A_m^*(P_m-S_m)/\max|E|^2$, followed by $a=F(E)$. It retains the source's approximate mixed-row selection and old-input best-save convention.

Power-PCGPA requires delays exactly equal to the Fourier time grid. Its explicitly labeled diagnostic linearly interpolates each observed frequency column from actual delays onto that grid, fills outside support with zero, and runs on this altered comparison data only. Projected signals are reversed in their delay axis and each time-column $k$ is cyclically rolled by $k$. If the resulting matrix is $C$, one power update is $v=\overline{C}E$, where the overline denotes elementwise complex conjugation rather than matrix transpose, followed by $E=\overline{v}/\|v\|_2$ and $a=F(E)$. Both the regridded error and the returned field's forward error on the original actual delays are reported. Primary COPRA and PIE never use the regridded observations. The source's ineffective PCGPA input-axis check is replaced by an explicit comparison-grid guard.

A separate global-step sensitivity diagnostic initializes all alpha variants from the same freshly computed primary COPRA best spectrum, explicitly forces global mode, and runs200 global updates per variant. It saves each mode array and asserts200 global/zero local steps, so alpha is actually active; this is not mislabeled as a standard cold-start trajectory.

The paper's comparisons of COPRA, PIE and PCGPA are retained for this SHG-FROG acquisition. Its broad survey of other nonlinear schemes and the expensive generic LM comparison on separate simulated examples are outside this bound512-point experimental task; no missing method is fabricated.

## Output and evaluation

The returned envelope is divided by $\sqrt{d\sum|E|^2}$ to give unit discrete energy and units s$^{-1/2}$. This convention removes the unobservable overall amplitude before serialization; the Judge separately checks the submitted energy rather than accepting arbitrary amplitudes.

The hidden reference is an identified reproduced standard-method result, not actual physical truth. Evaluation compares all512 complex samples after global phase, a continuous periodic time shift and optional conjugate time reversal. No weak-tail mask or magnitude-only substitution is used. Continuous alignment maximizes complex spectral cross-correlation: a dense oversampled Fourier scan selects a bracket and bounded scalar refinement determines the shift; the complex overlap determines the phase. Unit-energy complex relative L2 error and coherent overlap use the same alignment. Observation consistency recomputes SHG physics on the actual delays with a single fitted positive gain. Nonzero quality limits are frozen from independent original seeds/initializations and controls before the cleaned full reconstruction is scored; exact parity is not interpreted as physical accuracy.
