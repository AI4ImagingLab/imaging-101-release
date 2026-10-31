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

## Historical v1 reproduction diagnostics and limitations

The original extraction used standard-method pixel NCC>=0.999999 and dynamic-range NRMSE<=1e-5 as numerical reproduction gates. They are historical v1 diagnostics, superseded for benchmark acceptance by the complete observation objective below. The original algorithm's 1e-12 count floor is numerical RL arithmetic only; the v2 physical forward has no added count/noise floor. Original source-count closure <=0.001 remains. No actual sky truth is available.

Finite Monte Carlo response and averaged-attitude conversion are approximations. The adopted fixed-background Poisson objective retains the published count-fitting convention, without claiming an exact shifted-Poisson conditional sampling law for a known realized background. Descriptive overdispersion or deviance is not a calibrated p-value.


## Current benchmark contract v2 (2026-10-29)

The current v2 benchmark evaluates a nonnegative measurement-consistent conditional sky, not unique spatial truth. All 51,240,960 bins enter independently evaluated float64 factorized Poisson deviance; the fixed attainable boundary is 2.729057677375829 (measured original-field value 2.7290566773758287 plus predeclared 1e-6). Original source-count closure <=0.001 is unchanged. No response/noise/prior is invented. Response tallies actually have uint16 dtype and maximum1158, so conversion to uint8 would be lossy. The evaluator streams the exact integer array into a read-only temporary memmap and converts only 2048-column float64 tiles, enforcing at least6GiB available-host headroom. It imports no solver and never reads a private forward cache. Unsupported positive-count/zero-mean bins produce a finite reporting penalty and quality fail1, not NaN/Inf.

The original30-RL algorithm, raw observations, stored output and method baseline are unchanged. Historical pixel NCC/NRMSE and30-step comparisons remain explicitly labeled v1 reproduction diagnostics, not current pass gates. A scalar gain profile was essentially stationary (improvement only6e-14); it is not treated as meaningful new inversion evidence. An independent one-dimensional contrast direction with complete-forward line search lowers normalized deviance by4.455e-5 while preserving original source counts; an actual standalone full-data recomputation accepts it. Flux-matched wrong/flat maps are rejected by deviance despite exact count closure. This is a conditional improvement control, not a fresh blind reconstruction.

Full commands, failed memory-guard diagnostic, current tests, full-bin controls, independent scalar contractions and final hashes are recorded in the external benchmark_readiness_20261029/compton_v2 archive. Historical full active30-update notebook evidence remains valid for algorithm reproduction; only current evaluation and explicitly historical diagnostic cells are re-executed for this semantic revision.
