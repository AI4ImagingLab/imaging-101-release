# Implemented ordinary Poisson reconstruction

The selected method is the standard fixed-background Richardson–Lucy equation in Siegert2020 Section4.1 Eq7. The actual two complete upstream runs used the unchanged `RichardsonLucyAdvanced` class with every optional flag false because the capture passed a plain mapping rather than its nested configuration wrapper. Those runs are explicitly identified as ordinary RL. The intended advanced variant was not completed: corrected configuration exceeded the resource budget during its reachable before/after expectation copies. No advanced reconstruction or metric is substituted in comparisons.

## Unknown, units and discretization

The unknown vector $x$ contains3072 Galactic HEALPix RING nside16 intensities, each integrated over509–513keV, in photons cm$^{-2}$ s$^{-1}$ sr$^{-1}$. The measured count array has278 attitudes and184320 detector bins, the product of one energy bin,60 Compton-angle bins and3072 scattered-direction pixels. All bins are retained. No spatial mask or image alignment is used.

For attitude $a$, Galactic pixel $j$, local incoming direction $k$ and measured bin $q$, the sparse conversion weight $C_{ajk}$ is in seconds. The detector response is $R_{kq}=sN_{kq}$, where $N$ is the exact uint16 Monte Carlo tally and $s$ the public float32 area factor in cm² per tally. The pixel solid angle is $\Omega=4\pi/3072$ sr. The count operator and its ordinary transpose are

$$
(Ax)_{aq}=\Omega\sum_kR_{kq}\sum_jC_{ajk}x_j,\qquad
(A^Ty)_j=\Omega\sum_{ak}C_{ajk}\sum_qR_{kq}y_{aq}.
$$

The public exposure $e=A^T\mathbf1$ is independent of the sky target and is in cm² s sr. A separate local computation of this calibration sums each response row first, sums sparse conversion weights over attitude, and contracts the two factors. Its full-array parity is tested. The original HDF5 coordinate records include one underflow slot per axis; preprocessing subtracts one from each of the three coordinate rows. No nonzero conversion weight lies in an overflow bin.

The exact background realization $b$ is injected into the supplied event count array $d$ and known here. This ideal condition is explicit, rather than treating a second noisy measurement as an exact expectation. The source count residual $d-b$ is nonnegative on this bound acquisition. Background normalization remains one throughout reconstruction.

## Objective and updates

The native count expectation uses a fixed numerical floor $\eta=10^{-12}$ counts:

$$
\mu=Ax+b+\eta,\qquad
\ell(x)=\sum_{aq}d_{aq}\log\mu_{aq}-\sum_{aq}\mu_{aq}.
$$

The omitted log-factorials are constant with respect to $x$. This is an unnormalized total log likelihood, without a per-bin divisor. The floor is not part of the linear transpose or exposure definition. Initialize every component to $10^{-4}$ in the intensity units above. At iteration $n$, calculate

$$
y_{aq}=d_{aq}/\mu_{aq}^{(n)},\qquad
\delta_j=x_j^{(n)}\left[(A^Ty)_j/e_j-1\right],\qquad
x_j^{(n+1)}=\max(0,x_j^{(n)}+\delta_j).
$$

Preserve the additive delta/update order in float32; replacing it by an algebraically collapsed multiplicative expression can change cancellation and rounding. There is no exposure weighting of the delta beyond the denominator, angular smoothing, accelerated step, line search, prior penalty or background fitting. The step multiplier is one. Every sky pixel has strictly positive exposure, so no automatic zero-exposure mask is activated.

Recompute the expectation and likelihood after each update. Retain the source driver stopping condition: at most30 updates; after the second update stop if the likelihood increment is nonnegative and below0.01. A negative increment does not trigger early termination. Both complete original runs reached30. Source and local likelihood arithmetic uses float32 full-array sums; scalar metric calculations use float64.

## Numerical storage and execution

Calibration tallies remain exact uint16 in memory. A forward output-column tile contains at most8192 measured bins, and a transpose output-column tile contains at most32 local incoming directions. Calibrate each tile to float32 immediately before its general matrix product. Each dot product retains its entire reduction dimension: all3072 incoming directions in the forward product, or all184320 detector bins in the transpose. This is a storage and batching choice, not a truncated response or new operator. Sparse attitude rotation is an explicit SciPy CSR product. The complete operator's forward, transpose and exposure agree with independent frozen source arrays to approximately$10^{-7}$ relative RMS, below the predeclared$10^{-5}$ gate.

The expectation buffer may be overwritten with the ratio only after its likelihood has been consumed. Intermediate images, deltas and log likelihoods are actual newly computed arrays. CPU execution uses one numerical-library thread. The cleaned runtime contains no Histpy, Healpy, COSIpy or other imaging reconstruction dependency.

## Evaluation conventions and limitations

The hidden comparison is a standard-method result, not actual sky truth. There is no alignment, cropping, field normalization or intensity rescaling. NCC is cosine similarity without mean subtraction; NRMSE is RMS error divided by the reference dynamic range. Before local inverse extraction, numerical boundaries were frozen at NCC$\ge0.999999$ and NRMSE$\le10^{-5}$. The complete original independent rerun is bitwise identical; the nonzero tolerance allows ordinary float32 backend and contraction-rounding differences.

Source-count closure is $|e^T\hat{x}-\sum(d-b)|/\sum(d-b)$ and must be at most0.001. The native full-forward value is0.0004125728773. Zero, flat, permuted and0.95-scaled maps fail the frozen image criteria; scaling cannot be hidden by cosine similarity alone. Metrics for every required output must pass, with actual truth taking priority if an evaluator has independently supplied it.

Reference agreement does not establish physical truth. The finite Monte Carlo response and averaged-attitude conversion are approximations. Original full-data attitude residuals are overdispersed, and the fitted-mean sparse-Poisson deviance differs from a fixed-mean expectation. These are reported quantitatively, without a calibrated p-value or claim that every residual is explained by Poisson noise.
