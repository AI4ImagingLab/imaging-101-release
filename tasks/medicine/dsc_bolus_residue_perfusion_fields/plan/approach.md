# Implemented mathematics

## Binding, unknown and coordinate conventions

The real author GRE volume is retained without cropping/downsampling. Raw public axes are(batch,slice,row,column,time)=(1,22,128,128,80); inverse time-domain samples are160lags, interval1.55s, from0 to246.45s inclusive. The160-bin periodic extent is248s, not249.45s. In-plane and slice spacings are1.796875mm and4.399999618530273mm. Time origin is0. All calculations are float64 except the unchanged public int16 observations/boolean support.

Support89588 is fixed only from observations before original reconstruction; it excludes23otherwise-anatomical pixels with nonpositive recorded signals. All input intensities remain public. The entire support and every output lag are evaluated; outside support the declared output is zero. There is no spatial regularization, physiological clipping, magnitude conversion, reference alignment or fitted output gain.

Unknown k=F R is the flow-scaled impulse-response proxy(s^-1), not an independently measured retained-fraction truth. F is the source signed peak of k, not maxabs. Other required quantities are source leakage-corrected relative volume(dimensionless) and volume/F(s). Common tissue/arterial concentration proportionality, hematocrit and density factors are one. No absolute clinical perfusion claim is made.

## Signal transformation and baseline

For each nonempty slice compute its mean supported intensity at all80times. Begin baseline count p=3. If abs((mean(first p)−next)/mean(first p))<.05, increment p; otherwise decrement p and stop. Reaching12 also decrements to11and stops. Empty slices retain original count2. The corrected source stride uses nRow*nCol per slice; the original used nSlice and therefore mixed spatial slices. No other baseline behavior is changed.

For each supported voxel S0 is its mean of the first p raw intensity samples. Compute c=−log(S/S0)/TE, with TE=.025s and proportionality1. Baseline S0 has ADU units, c inverse seconds. Strict positive finite observed signal is required on support; invalid logarithms are errors, not floored values. The original positivity support is acquisition input, not selected from reconstruction quality.

## Explicit acquisition discretization

Let a be the80-entry arterial proxy(s^-1), n=80, dt=1.55s. Circular G has160rows/columns and first column b: b0=a0; bi=(a(i−1)+4ai+a(i+1))/6 for1<=i<=78; b79=(a78+4a79)/6; b80=a79/6; remaining entries0. Gij=b((i−j)mod160). Forward predicted acquisition is the first80entries of dt Gk. Zero padding refers to the80unobserved data entries used in the inverse, not invented measured truth.

Causal sSVD comparison instead uses80×80 lower-triangular Toeplitz with b0=a0,b79=a79 and the same interior smoothing. Its forward is dt Gk. These source endpoint conventions are deliberately distinct. Both operators have a direct Euclidean transpose adjoint, tested by independent scalar sums/inner products. Multiplication by dt is necessary dimensionally: (s^-1)*s*(s^-1)=s^-1.

## Singular inverses and actual stopping

Use SciPy general SVD with gesvd, G=U diag(s) Vh; s are descending nonnegative singular values. For cutoff tau invert only s>=tau*s0. The mathematical expression is k=V diag(inv(s)) U^T c /dt; observations are padded to160for circular methods. The original oSVD omitted/dt; qualification adds this factor after its existing matrix expression. Source unchanged/frame-scaled output is separately archived, not silently compared in inverse-second units.

sSVD cutoff=.2, cSVD=.1. Both source flow maps are max(abs(k)) over their80/160lags. Primary oSVD tries tau=.05,.10,...,.95. Each trial recomputes the truncated response. Define OI=sum(abs(kj−2k(j−1)+k(j−2)),j=2..159)/(160*max(k)). Stop at the first strict OI<.035; otherwise retain the.95trial. OI uses the signed maximum; no absolute peak substitution. Primary flow is max(k). There is no iterative initial guess, line search, stochastic seed or extra optimizer. Chunk4096controls only memory; each voxel is independently evaluated with the same cutoffs. Tests compare all selected stages and complete primary fields to original numeric fixtures/results.

The primary inverse is the source adaptive truncated-SVD family, not a positivity-constrained least-squares replacement. For a fixed cutoff its pseudoinverse is the spectral-filtered least-squares solution; adaptive OI chooses that filter per voxel. There is no independent target-fit hyperparameter selection.

## Relative volume and leakage regression

Let y be all supported N×80curves in s^-1 and t=dt*[0..79]. Source uncorrected volume v=trapz(y,t)/trapz(a,t). Its nonenhancing set uses baseline sample count floor(mean(all22slice counts)), sample SD(ddof1), and abs(median(last11values))<=2*SD. This differs from the primary paper's last10mean/1SD; it is preserved as the supplied implementation variant, not described as identical paper preprocessing.

Let m(t) be the nonenhancing-set mean, I(t)=cumtrapz(m,t) with I0=0, and A=[−I,m]. Solve A*[K2,K1]^T=y using general least squares(gelsd). Coefficient uncertainty uses sigma²=sample variance of the first min(all slice counts) samples, here2because the final empty slices retain count2; coefficient covariance is sigma²*(A^T A)^−1. Define percentage CVj=100sqrt(covjj)/abs(Kj). The correction is vc=v+1(CV2<100)*K2*trapz(I,t)/trapz(a,t). The source's global empty-slice effect is preserved and explicitly disclosed, not quietly corrected. K1 is dimensionless and K2 has s^-1 units.

Transit time is vc/F on support. Outside support source MTT0/0NaNs are explicitly represented by required zeros; no supported value is dropped. Negative vc/MTT estimates and negative residue lobes remain valid numerical submissions and visible limitations of this conditional estimator, not claimed physical negative blood volumes.

## Evaluation and comparisons

No patient truth is available. Hidden baseline is the whole qualified original, repeated independently with47numeric arrays/options bitwise equal. Only the two diagnosed lines changed. Before local extraction, freeze individual whole-array NCC>=.999999 and dynamic-range NRMSE<=1e−7 for all four required quantities. Nonzero numerical tolerance is justified by deterministic source reruns and independent linear-algebra rounding allowances; it is not self-comparison zero made into a zero threshold. Full observation relative L2<=.3119447203693109 equals1.05times original.2970902098755342. The per-voxel best constant residue gives.794398454618137 and zero1; no vacuous null passes. No crop/magnitude/scale/alignment is allowed.

Notebook actively computes whole sSVD/cSVD/oSVD, reports original versus fresh metrics, all required maps, full-lag curves, OI/cutoff statistics, leakage comparison, six fixed slices, four OI choices and four synthetic proxy-noise levels. Diagnostic noise is explicitly generated by local generator with PCG64seed124; it never changes the bound real task data. Relative-noise SD is a fraction of supported proxy dynamic range. Every plotted fresh output is computed; frozen original maps are named as method measurements. No stable-spline method, absolute truth error or unavailable clinical diagnosis is asserted.
