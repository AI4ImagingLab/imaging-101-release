# Implemented receiver-response mathematics

The unknowns are signed radial and transverse receiver responses of one real GE.WLF three-component record. This is an operational finite-band receiver response, not an elastic velocity model, calibrated interface depth, or independently measured truth. The source proxy is measured vertical ground velocity. All arrays and computations retain float64; FFTs are complex128. Source/sample units and convolution coefficient units are m/s and dimensionless respectively.

## Bound data and calibration

The native 20 Hz samples nearest the fixed UTC interval 2011-03-11 05:56:30–06:00:30 are retained, inclusively (4801 samples). This window was fixed before independent upstream capture. Count records originally include larger MiniSEED blocks; selection uses nearest-sample trimming. Calibration was completed once outside the cleaned solver: ObsPy removes the complete public per-channel StationXML response, output VEL, 60 dB water level, cosine frequency prefilter (.02,.03,2,4) Hz, default mean removal and 5% cosine edge taper. Public `waveforms` are those actual calibrated measurements; no hidden reconstruction or target enters this process. The original sensitivities differ and normalized transfer functions differ by 3.134%; dividing by gain alone was explicitly rejected. Calibration is acquisition preparation, not new synthetic generation.

## Preprocessing and coordinate convention

For each Z,N,E trace independently, least-squares linear detrending removes a fitted line against sample index. A symmetric Hann window of length 481 supplies the first/last 240 weights; the interior 4321 weights equal one. This is a 5% per-side finite-window taper. Apply a fourth-order Butterworth bandpass, corners .05 and 1 Hz, represented by second-order sections with zero initial filter state. It is causal, one pass, with no reverse filtering. SciPy's normalized digital corner frequencies are .005 and .1 of Nyquist. Retain all 4801 samples.

Let b=33.06338963947741 degrees, from station to Tohoku epicenter clockwise from north. R=−N cos b−E sin b and T=N sin b−E cos b, with Z unchanged. This orthogonal transform preserves squared velocity norm and fixes transverse polarity. R points away from the source. No free angle fitting or target-derived alignment occurs.

## Standard spectral inverse

Let N=4801 and L=4860 be the next 5-smooth FFT length according to `scipy.fftpack.next_fast_len`, matching upstream. Do not substitute the modern `scipy.fft.next_fast_len`, which chooses 4802 and changes the inverse. Zero-extend the finite traces to L. DFT uses exp(−2πikn/L), inverse uses exp(+2πikn/L)/L. Frequency f is signed Hz from `fftfreq(L,.05)`. S is FFT(Z), and D is FFT of a selected Z,R,T trace. Set source power P=|S conj(S)|, Pmax=max P and W=max(P,.05 Pmax) elementwise. The Gaussian low pass is G=exp(max(−.5(f/.5)^2,−700)); .5 Hz is the standard deviation, giving exp(−.5) response at .5 Hz. Delay Q=exp(−2πif·30 s) places zero lag at sample600. Compute

```math
u_d[n]=\operatorname{IFFT}\left(GQ\,D S^*/W\right)[n],\quad
c=1/\max_{0\le n<N}u_Z[n],\quad h_d[n]=\operatorname{Re}(c u_d[n]).
```

The original complex numerical normalization is preserved before taking the real part. Imaginary amplitudes on these data are below 1.1e−16 and are roundoff, not a complex physical output. Both horizontal components use the same c (approximately62.172, exact measurement recorded privately). Truncate to firstN samples, retain negative responses, and save batch,R/T,lag. The source proxy self-response supplies normalization only; it is not an unknown submitted array.

The inverse is a closed-form stabilized source division, so there is no initialization, line search, iterative stopping criterion or extra regularizer. Water-level clipping bounds spectral amplification; it is not identical to a ridge least-squares objective, and no such equivalence is claimed. The underlying acquisition is a discrete convolution with no dt multiplier. Finite FFT boundary handling induces circular convolution on L points after zero extension.

## Actually computed comparisons

Time-domain spiking inverse builds A, a symmetric Toeplitz matrix whose first row is the source autocorrelation divided by its zero-lag value, then adds .1 to the zero-lag entry. The right side is the source-response correlation, with the upstream asymmetric zero padding needed to produce N lags and integer shift int(30·20−N//2)=−1800. Solve A x=b via SciPy general Toeplitz linear algebra. There is intentionally no source-energy division of the right side, matching rf; normalization by the absolute maximum source self-result cancels its common scale. Both component amplitudes share that normalization.

The iterative comparison is the Ligorria-Ammon sparse-spike update. Use L2=9720 (5-smooth next length ≥2N), dt=.05 s, Gaussian .5 Hz, source spectrum Sg=FFT(Z)G and source energy Es=sum IFFT(Sg)^2. Filter each response to dG. Initialize residual r=dG and spikes p=0, old normalized squared error=1. Cross-correlate residual and Sg, divide by Es·dt; mute lag indices N through L2−600−1, retaining the negative lags allowed by upstream `mute_shift=False`. Add the signed largest absolute correlation to p at its index. Recompute predicted filtered response IFFT(FFT(p)Sg)·dt and residual. Stop when the absolute percentage-point reduction in sum(r²)/sum(dG²) is ≤.001 or 400 iterations have been performed. Apply Gaussian and30s delay to the spike series, truncate toN, and normalize all three traces by the absolute maximum source result. Both horizontal comparisons reach the 400-iteration cap on this event; they are labeled capped, not converged. No optimizer substitution occurs.

Sensitivity experiments rerun the standard inverse at alpha=.005,.05,.2. Notebook also recomputes the bandwidth effect at sigma=.25,.5,1 Hz and removes every other time sample to test acquisition reduction with a separately defined10Hz inverse. The latter is an experiment only and does not change the native benchmark. No fresh method is replaced with an archived result.

## Evaluation and limits

Standalone Judge checks each component separately, with uncentered cosine similarity and RMS error divided by that component reference range. Actual truth is absent; the original exact-data standard-method result is a baseline reference. Repeated original runs are numerically identical; nonzero nrmse≤1e−6 and ncc≥.999999 allow numerical precision differences, including float32 submissions. They are method-agreement tolerances, not geological-accuracy thresholds. No crop, amplitude fit, translation, magnitude conversion or sign flip occurs. Full traces, including edge effects, are judged.

Reconvolution undoes the30s delay and c, convolves the declared finite horizontal result with measured Z on the4860 grid, and compares against the Gaussian-filtered measured horizontal traces. Its residual includes spectral water-level attenuation, finite trace truncation and model mismatch. It is reported in m/s and relative L2 units; it is neither disguised as ground truth nor required to equal a hypothetical Gaussian chi-square. A single finite record cannot reliably separate all structural conversions from source-proxy error, scattering, noise and late arrivals. No depth inference is made.
