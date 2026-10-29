# Implemented rainfall reconstruction

## Binding and physical units

The complete original48-hour Linkdata contains496180 rows before the input-only consistency checks. These are actual received-power extrema, not a simulated object. The output is accumulated rainfall H in mm on all38063 native projected grid points over the96 interval ends201109100815 through201109110800. The entire preceding history remains available for baseline inference. No final estimate selects the target support and no grid point is dropped. There is no independent rainfall truth: the selected original complete wet/dry method supplies a conditional reference.

Each record has frequency f(GHz), integer UTC timestamp, Pmin and Pmax(dBm), link length L(km), antenna WGS84 positions, numeric ID and optional polarization. The native polarization is unavailable, so the vertical calibration is assumed. Powers are summarized from10Hz sampling over15min, with nominal1dB quantization. The supplied projected coordinates are the known EPSG4326→EPSG28992 transformation divided by1000, in km. This input-only geometric calibration uses no rainfall estimate.

## Preprocessing and chronological discretization

Restore missing values from the explicit mask, sort stably by timestamp, retain frequencies12.5≤f≤40.5GHz, and process IDs in first-appearance order. Remove every repeated timestamp for a given ID. Remove an entire link if frequency, length or any antenna coordinate varies within the supplied period. Exclude records with missing values in the first ten columns; missing polarization is permitted. This reproduces465514 original valid records from2472 link IDs over192 interval ends. Only this input-based consistency selection precedes the inverse calculation.

Use arrays indexed by sorted time t and first-appearance link index i. The interval duration is delta=0.25h. A preceding24h window including the current observation contains at most96 samples: indices max(0,t−95) through t. Timestamp zero is never treated as a physical epoch. The selected native sequence is uniform; the first time has no rolling estimate.

## Wet/dry classification

Let B(t,i) be the maximum Pmin within the preceding24h window, defined only when at least24 valid samples(6h) exist. Define d(t,i)=Pmin(t,i)−B(t,i), in dB, and e(t,i)=d(t,i)/L(i), in dB/km. Two links are neighbors only when all four distances between their start/end pairs are strictly below15km. The selected link counts as a neighbor when its own length satisfies this condition.

Compute neighborhood medians m(t,i) of d and n(t,i) of e only when at least3 finite neighboring d values exist. The initial dry indicator D is0(wet) if both m<−1.4dB and n<−0.7dB/km,1 otherwise, and unavailable when either median is unavailable. When D=0 and the link's own d<−2dB, also label its previous two and next one time samples wet. The initially unavailable D locations remain unavailable. All inequalities and endpoint conventions follow the selected source.

The outlier statistic is

```math
F(t,i)=\delta\sum_{u=\max(0,t-95)}^t[e(u,i)-n(u,i)],
```

with unavailable summands omitted. Its units are dB·h/km. It is evaluated only where the current neighborhood count suffices. The original R function reuses the F vector across link loops, so unsupported entries can retain a previous link's values; the local implementation preserves this diagnostic convention explicitly. These entries have unavailable D and therefore cannot produce rainfall. This is a source diagnostic artifact, not a physical value or an extra observation. Complete stage tests compare the actual original arrays, and unsupported-D implies unavailable rain independently.

R sum uses extended-precision accumulation. On this Windows runtime, math.fsum provides faithful compensated accumulation before multiplying by0.25. Ordinary prefix sums introduced a rounding-only decision mismatch at an exact−32.5 threshold in one historical sample; the captured failure is retained privately. No threshold padding, altered comparison or discarded sample is used.

## Reference power, correction and nonlinear inversion

For dry intervals form M=(Pmin+Pmax)/2 in dBm. The reference Pref(t,i) is the median finite M in the same preceding24h window, provided at least10 dry samples(2.5h) exist. For the no-wet/dry comparison all intervals may contribute to this reference.

The original outlier filter makes Pmin unavailable wherever F≤−32.5dB·h/km; it preserves table rows and Pmax. The corrected minimum power equals observed Pmin only when that value is below Pref and the interval is wet, and otherwise equals Pref. It is unavailable if Pref, Pmin or D is unavailable. The corrected maximum equals observed Pmax only when corrected Pmin<Pref and Pmax<Pref; otherwise it equals Pref. It is unavailable when the corrected minimum or observed maximum is unavailable.

Define Amax=Pref−PminCor and Amin=Pref−PmaxCor, both in dB. Interpolate the supplied a and b calibration tables linearly against log frequency, separately by polarization; unknown polarization uses vertical coefficients. With independently calibrated antenna correction Aa=2.3dB and mixing coefficient alpha=0.33, evaluate

```math
R_{\min}=a\left[\frac{\max(A_{\min}-A_a,0)}L\right]^b,
\quad R_{\max}=a\left[\frac{\max(A_{\max}-A_a,0)}L\right]^b,
\quad R=(1-\alpha)R_{\min}+\alpha R_{\max}.
```

R is mm/h, b dimensionless and a has units(mm/h)(km/dB)^b. Missing Amax produces missing R, not a dry measurement. This is an empirical inversion of attenuation extrema; it does not recover the full temporal power waveform or varying intensity along an individual link.

## Midpoint observation approximation

For each interval, retain finite R and assign it to the arithmetic midpoint of the projected antenna endpoints. Average all observations with exactly coincident midpoints, including opposite link directions. Preserve the first-appearance coordinate order. This is the selected source's point approximation to line observations; it is not an exact line-integral tomography model.

## Conditional ordinary kriging

Let x(i) be unique midpoint positions and v(i) their intensities. At every grid point q select the50 nearest midpoint observations in Euclidean km, using all when fewer than50 exist. A spherical semivariogram gamma has gamma(0)=0 and, for positive h,

```math
\gamma(h)=C_0+C\begin{cases}1.5h/\ell-0.5(h/\ell)^3,&h<\ell,\\1,&h\geq\ell.\end{cases}
```

The range ell is in km and C,C0 in(mm/h)^2. With J the day of year of the interval end and D=0.25h expressed as its numerical hour value,

```math
\ell=10^{-3}[15.51D^{0.09}+2.06D^{-0.12}\cos(2\pi[J-7.37D^{0.22}]/365)]^4,
```

```math
C=[0.84D^{-0.25}+0.20D^{-0.37}\cos(2\pi[J-162D^{-0.03}]/365)]^4,
\qquad C_0=0.1C.
```

These independently calibrated empirical coefficients extrapolate the published hourly-to-daily gauge climatology to15min. J changes at midnight, even within the required accumulation window. No covariance is estimated from a hidden rainfall target.

Form Gamma(i,j)=gamma(norm(x(i)−x(j))), g(i)=gamma(norm(x(i)−q)), and solve the explicit saddle equations

```math
\begin{bmatrix}\Gamma&\mathbf1\\\mathbf1^T&0\end{bmatrix}
\begin{bmatrix}w\\\lambda\end{bmatrix}=\begin{bmatrix}g\\1\end{bmatrix}.
```

The vector1 contains ones and lambda is the unknown-mean constraint multiplier. The interpolation is r(q,t)=max(0,sum(i)w(i)v(i)). There is no extra fitting objective, iterative optimizer, spatial regularizer or stopping tolerance: this selected conditional inverse uses direct general linear-system solves. Chunk size256 changes only memory allocation. Distances have no periodic wrap and no extrapolated boundary rainfall is prescribed. The complete daily output is H(q)=0.25sum(t)r(q,t), over all96 native target intervals.

The alternative IDW comparison uses every finite midpoint observation with weights proportional to inverse squared distance, normalized to sum one. A query coincident with an observation receives its exact value. This is an independently computed comparison, not the required daily estimator.

## Source qualifications and evaluation

The exact selected source is RAINLINK1.31,commitce355d68dcb59ec17f75d05babc4a6f6d2ee90f0. The supplied coefficient text files have a literal f a b header, while Run.R originally reads them without header=TRUE. Two explicit header reads are the only numerical-input API qualification; all coefficients were independently verified bitwise. Complete original Run.R also executes mutually alternative examples sequentially and overwrites its earlier wet/dry reference and correction. The main binding therefore selects the documented complete wet/dry option consistently, before inspecting its field, using unchanged original domain functions. The actual successful full Run.R no-wet/dry result remains a scientific comparison, not the selected main baseline. No claim of an entirely unchanged script is made.

The whole original main field is a method reference, never true rainfall. Two independent complete source runs match exactly. Before any local solver, the range-NRMSE boundary9.663465651076436e−9 was frozen as1.5 times the actually measured float32 serialization error6.4423104340509565e−9. Independently solved full spatial stages agree with original gstat within2.170708057747106e−12mm/h. This narrow reproducibility bound does not quantify physical rainfall uncertainty. Zero, uniform and shuffled fields have range-NRMSE0.2540,0.1828 and0.2581. No field mask, output rescaling or time/space alignment is allowed. Nonnegativity is independently required on every grid point.

The standalone Judge always prioritizes actual truth if an actual-truth asset is supplied; malformed truth cannot fall back to the method reference. It rejects missing keys, wrong shapes/dtypes, nonfinite arrays, invalid threshold types/ranges, degenerate references and overflowed metrics. All scored values must be finite and JSON-safe.
