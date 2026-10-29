# Implemented spherical deprojection

## Benchmark v2 physical output and acceptance

The historical `onion_peeling` stage and its upstream parity remain unchanged. `reconstruct` now reports the final nonnegative hydrogen density and its exact square as emissivity; the independently median-smoothed historical emissivity is retained as `historical_smoothed_emissivity` diagnostics. This corrects two formerly inconsistent representations of one physical field. Negative final densities raise instead of silently entering the physical submission. This is an explicit output-semantic revision, not unchanged upstream emissivity reproduction. Existing raw measurements and upstream stage fixtures are unchanged; the old bundled output, method baseline and v1 metrics are privately archived under `benchmark_readiness_20261029/a3158_v1_archive`.

The v2 Judge uses no hidden reconstruction: both representations must be nonnegative, satisfy relative q=nH² error <=1e-10, and pass the original unchanged 35-annulus forward RMS limits (density 1.0610366206199466; emissivity 1.0636229762545693). These are count-uncertainty consistency criteria, permitting scientifically indistinguishable profiles without requiring an MC seed or smoothing trajectory. The complete exterior formula is public in README. Historical reference-agreement metrics below remain reproduction evidence only.

The selected method is the author's `Deproject.OnionPeeling`, not the multiscale King-profile/HMC alternative. The full native count, exposure and instrumental background maps are retained. All computation is CPU float64. There is no PSF deconvolution, ellipticity, free temperature profile, or adjustable spectral conversion.

## Observations and background

Set count pixels with exposure <=0 to zero only for finding the center. Apply a Gaussian with sigma5 pixels and SciPy default reflect boundaries. Average the coordinates of tied maxima; this image gives zero-based x271,y280. Radius is Euclidean pixel distance times0.2491218 arcmin. Keep exposure >0.05 maximum. For contiguous annuli assign lower-inclusive/upper-exclusive rounded boundaries (five decimals plus0.000005 arcmin), with zero lower boundary in the first annulus.

Observed area is the number of valid pixels times pixel area. Surface rate is sum((count-instrument_background)/exposure)/area. Statistical standard deviation is sqrt(sum(count/exposure²))/area. This treats the supplied instrumental background as a fixed calibration. Build135outer annuli of width1/3arcmin from0to45. In centers30to40arcmin fit constant b by minimizing sum((rate-b)/sigma)². Its exact solution is sum(rate/sigma²)/sum(1/sigma²), with standard error1/sqrt(sum(1/sigma²)). The original log-parameter MINUIT objective has this same optimum. Default original optimizer tolerance and a tight1e-9 tolerance are preserved separately in provenance; the local closed-form optimum matches the tight run. Subtract b from the35selected logarithmic-bin rates and add its variance to every bin error. As in the source, common-background covariance is ignored.

## Geometry and calibration

Let u,v be shell inner/outer radii in cm, U,V annulus inner/outer impact parameters, and P(t)=max(t,0)^(3/2). The projection volume is

Vij=4pi/3 [P(vj²-Ui²)-P(vj²-Vi²)-P(uj²-Ui²)+P(uj²-Vi²)].

It is upper triangular for contiguous shells. The implementation preserves the normalized-ratio arithmetic of upstream `MyDeprojVol`, including assigning each nonfinal outer radius to the next inner radius, to avoid thin-shell cancellation differences. The independent Judge uses the expanded positive-part expression in angular units before unit conversion. Shell radii use70.73126033926604kpc/arcmin and3.0856776e21cm/kpc.

Define eta=ne/nH=1/0.8527, z=.059, DL=272.69501551950657Mpc, DA=DL/(1+z)². XSPEC normalization is1e-14 integral(ne*nH)dV/[4pi DA²(1+z)²], and count rate=43.53*normalization. Thus K=4pi1e14 DL²/[(1+z)² eta43.53] converts count rate into integral(nH²)dV. This derivation establishes that upstream `dens` is hydrogen density, despite its electron-density prose. No numerical convention is changed. Spectral calibration uses the author's documented5keV, NH=.0118e22cm^-2, aspl, PSPC gain2 response and0.42-2.01keV assumptions. It is a known fixed conversion, not estimated from the hidden profile.

## Finite outer boundary

For angular inner/outer radii u(i),v(i), define a=u(last),b=v(last). First compute

t(i)=(a+b)ab/v(i)^3 *[-1+2/pi*(2v(i)/b+acos(v(i)/b)-v(i)/b*sqrt(1-v(i)^2/b²))].

For u(i)>0 replace this by

t(i)=(a+b)ab/[(u(i)+v(i))u(i)v(i)] * [1-2/pi*(v(i)/u(i)*acos(u(i)/b)-acos(v(i)/b)-v(i)/b*(sqrt(1-u(i)^2/b²)-sqrt(1-v(i)^2/b²)))/(v(i)/u(i)-1)].

With full annulus area A(i)=pi(v(i)²-u(i)²) and emission e(i)=rate(i)*K*A(i), the source dimensionless correction is c(i)=t(i)*A(i)/A(last)*e(last)/max(e(i),1e-10). The corrected emission is e(i)(1-c(i)). All35bound rates are positive. Equivalently its exterior surface-rate contribution is t(i)*rate(last); this latter expression is independently used in the Judge. This is a prescribed exterior model, not measured exterior truth.

## Inversion and uncertainty

Use1000Gaussian realizations: e_sample=e+sigma_e*Z, where sigma_e=profile_error*K*A, and Z uses legacy MT19937 seed24713 in shape35x1000. This seed controls uncertainty integration only. Solve V*q_sample=e_sample*(1-c) with a general dense linear solve. This is direct onion peeling; no iterative optimizer or fabricated iterations are used. Take the median across samples then apply the source five-neighbor median filter. At the second and penultimate positions use four valid neighbors; first/last use the median of the endpoint, its neighbor, and the extrapolation3*endpoint-2*neighbor. Return this smoothed q as emissivity. Return the same filter applied to sign(q)*sqrt(abs(q)) as hydrogen density. Uncertainty is0.5/sqrt(abs(q))*std(q_sample). No clipping of physical outputs is added.

The implementation has no regularizer objective, step size, line search, or convergence tolerance. Median smoothing and the fixed exterior assumption supply its stabilization. Optional notebook comparisons disable exterior correction or smoothing without changing the benchmark method. A second MC seed and five independent original MC reruns expose numerical integration variation.

## Evaluation

Both35-element outputs are evaluated: cosine NCC without mean subtraction, and RMS error normalized by reference dynamic range. The baseline is the independently run original method, not truth. Bounds were fixed from five additional original1000-sample MC runs with unchanged observations and method:1.1times worst NRMSE, and1minus1.1times worst NCC departure. This admits ordinary MC integration changes without requiring the private seed. Independently recomputed observation residual RMS, standardized by count/background uncertainties over all35annuli, must also pass for both q and density². Bounds are1.1times worst original-run RMS. The standalone Judge prioritizes actual truth if supplied, validates every required finite boundary and NPZ array, and returns error/empty metrics for malformed inputs. There is no fitted rescaling, alignment, or omitted shell.
