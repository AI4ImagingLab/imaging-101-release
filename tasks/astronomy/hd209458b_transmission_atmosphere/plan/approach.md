# Selected isothermal transmission and finite ensemble protocol

The reference is the original author v4 example on all42measured HD209458b bands. It is not a retrieval of2020PLATONII's HD189733b data. The2019 paper describes an older2.1release; v4 preserves the published physical family with the author's current250layer discretization. Treat this distinction as version provenance, not an identical historical figure reproduction.

## Atmospheric physics and numerical convention

All quantities SI except temperature K, atomic masses u, and base10logparameters. Reference radius at1e5Pa. Isothermal250point log-pressure profile between supplied first/last pressure nodes, strict P<10**log_cloudtop_P opacity visibility. Equilibrium abundances are bilinearly interpolated in logZ/log10CO at CO=.53, exponentiated and floored1e-99. Each species' logabundance is linearly interpolated over T/log10P using RectBivariateSpline(kx=ky=1). Mean molecular mass mu=sum(mixingratio*speciesmass), original order, without renormalization.

Spherical hydrostatic integration in logpressure obeys d(1/r)/dlnP=kBT/(GMp*mu*AMU), using trapezoid averages of thermal and mu splines at segment endpoints. Separate inward/outward integrations start from1bar reference radius. Returned radii descend, layer width dr=-diff(r). Original Hill-radius check uses estimated radiative-equilibrium distance Rs*(Tstar/T)^2 and Mp/(3Msun)^(1/3); retain this implementation rather than substitute another atmospheric escape model. Unbound atmospheres are rejected.

At each retained enclosing calibration T/P node: alpha_gas=sum_s opacity_s*mixing_s; alpha_Rayleigh=10**logscatter*(128*pi**5/3)*n*sum_s(mixing_s*polarizability_s**2)/lambda**4; alpha_CIA=sum_pairs coeff_pair*n1*n2. Number density n=P/(kBT). Sum active species/pairs in original order. Interpolate total alpha linearly in T/P (not logalpha) onto atmospheric profile. Gas coefficients have units1/m per mixingratio, polarizabilities m3 and CIA coefficients m5 by independent dimensional derivation. Dimensional labels should not be overstated as separately measured conversion calibration.

Chord geometry: for descending shell boundaries r_j and impact radius r_k, path matrix is the difference of 2sqrt(max(r_j**2-r_k**2,0)) across neighboring boundaries. Midpoint alpha=(alpha_j+alpha_{j+1})/2, tau=alpha_mid.T @ path_matrix. Dimensionless absorption fraction is1-exp(-tau), without a replacement stable-expm1 convention. Depth=(minradius/Rs)^2+2/Rs^2*sum(absorptionfraction*r_lower*dr). Band means include lambda>=lower and<upper and supplied stellar weights. No extra instrument convolution or hidden band adjustment. Preserve numerical constants in physics_model.py.

## Prior, likelihood and sampler

Eight parameters are [Rs,Mp,Rp,T,logscatter,logZ,logcloud,errorfactor]. Rs and Mp have Gaussian priors centered1.19Rsun/.73Mjup, sigma.02Rsun/.04Mjup. Remaining strict bounds: Rp>0,300<T<3000,0<logscatter<5,-1<logZ<3,-.99<logcloud<5,errorfactor>0. Priors are normalized Gaussian densities for first2parameters; uniform components contribute zero logdensity inside support, -inf outside. Gaussian likelihood is -.5*sum(((prediction-data)/(factor*nominalerror))**2+log(2*pi*(factor*nominalerror)**2)). No dropped normalization, squared-error-only replacement or silently enlarged data errors.

Initialization is walker-major RandomState(seed) uniform draws for49walkers; walker0 is originalfiducial [1.19Rsun,.73Mjup,1.4Rjup,1200,0,0,4,1]. Rs/Mp uniform +/-2sigma; Rp .9..1.1fiducial; T600..1800K; logscatter0..1; logZ-1..3; logcloud-.99..5; errorfactor.5..5. First walker has a prior-boundary scattering guess0 and therefore initially -inf posterior, as upstream; no hidden repair. Benchmark seed20261028is maintainer reproducibility only, not hidden target data; independent seed20261029originalfullrun tests finite-sampling variation.

Exactly50walkers and1000transitions, no early stopping/burn discard/thinning in summary. Each transition consumes the original singlemove-selection RNG draw, shuffles two red/blue groups and applies a=2 stretch proposal q=c+z*(x-c) with z=(U+1)^2/2 and randomly selected complementary walker c. Log acceptance difference is7log(z)+logpost(q)-logpost(x), accepted if>log(U). Each half update uses current complementary group including earlier-half updates. All draws, group ordering and strict comparisons match emcee3.1.6. Store chain(walker,step,parameter), logp(walker,step) and perwalker accepted/1000. No ready-made sampler dependency. Median uses all50000samples; best is maximumlogp sampledpoint; predicteddepth is a new forward computation at best.

## Calibration reduction and evaluation

Only CO=.5,.6 nodes retained because fixed.53interpolates exactly between them in logCO. Removed unused100Kchemistry row contained NaNs and is never queried for T>300. Every original activeopacity species,collisionpair, pressure node and bound wavelength is retained. Exact-originalclass comparisons at9independentT/logZstates show every active intermediate unchanged; this is scientific-equivalence evidence, not a smaller substitute model.

No physicalGT. Standard-methodreference is full original50x1000run, primaryseed20261028. Independent second originalseed medians/best/spectrum and physicalmetric differences validate preselected stochastic boundaries. Parameter comparisons use originalfullchain std(ddof1) per each8coordinate; max standardizedmedian<=.5 and best<=3; all42depthmaxdifference<=100ppm; unscaled observationchi2<=275.9259882866698; normalized logposterior>=251.249041852333. These are fixed before localfullresults; do not relax to absorb nonconvergence. Independent forward consistency between submittedbest and spectrum<=.0001ppm is a float64contractcheck, not experimentalaccuracy. Fitted errorfactor~2.4is a published unknown, not a posthoc threshold adjustment: report nominal and scaledchi2 separately. Originalnominalchi2~250.84versus scaled43.39for42bands motivates honest heterogeneity analysis.

Notebook actively repeats all50x1000transitions after tests/plans are aligned. It retains originalbestfit/whole8parametercorner scientific comparisons; adds fullwalker traces, prior/noise diagnostics, posterior correlations/quantiles, predictivebands, hydrostatic/opacity/tau structure and gas/Rayleigh/CIA forward ablations. These latter are freshly evaluated at selectedbest and are not alternative retrievals. Neither chain length nor convergence is inflated by labeling correlatedwalkers independent samples.

## Mathematical definitions for independent review

Let pressure be P in Pa, radius r in m, temperature T in K, planet mass M in kg, and mean molecular mass mu in atomic mass units. Constants G, k and m are the gravitational, Boltzmann and atomic mass constants, respectively. The reciprocal-radius integration used above is

```math
\frac{\mathrm{d}(1/r)}{\mathrm{d}\ln P}=\frac{kT}{GM\mu m},\qquad
\frac{1}{r_j}=\frac{1}{R_p}+\sum_{l=1}^{j}\Delta\ln P_l\frac{k\overline{T}_l}{GM\overline{\mu}_l m}.
```

R-p is the specified one-bar radius. Index l enumerates pressure segments and j the integrated endpoint; overbars denote arithmetic averages of the two spline endpoint values. Negative pressure increments above one bar therefore increase radius, as required physically.

Let f-s denote number mixing fraction of species s, n=P/(kT) the total number density, a-s its polarizability volume in cubic metres, K-s its supplied gas absorption coefficient per unit mixing fraction, and C-st the supplied collision coefficient in m to the fifth power. With scattering enhancement S, wavelength lambda in m and pair species t, the implemented extinction is

```math
\alpha(\lambda)=\sum_s f_s K_s(\lambda)
+S\frac{128\pi^5}{3}n\lambda^{-4}\sum_s f_s a_s^2
+\sum_{s,t}C_{st}(\lambda)(f_s n)(f_t n).
```

The last sum follows the supplied pair list exactly; it does not add symmetric pairs or an extra factor of two. Each term has units inverse metres. Midpoint extinction times shell chord length gives dimensionless optical depth. The radius quadrature is the lower-boundary rule, with positive layer width and descending radii, so opacity increases blocked area.

For observation index i, measured depth d-i, predicted depth D-i, nominal uncertainty sigma-i and positive inferred error factor e, the likelihood is

```math
\log L=-\frac12\sum_{i=1}^{42}\left[\frac{(D_i-d_i)^2}{e^2\sigma_i^2}+\log(2\pi e^2\sigma_i^2)\right].
```

The normalized two-coordinate Gaussian prior is added to log L. For dimension d=8, current walker x, complementary walker c and stretch factor z, the proposed vector q and acceptance condition are

```math
q=c+z(x-c),\qquad z=\frac{(U+1)^2}{2},\qquad
\log V<(d-1)\log z+\log\pi(q)-\log\pi(x).
```

U and V are independent uniform draws on [0,1), and pi is the posterior density in the declared SI/log coordinates. Invalid states have negative infinite log density. The shuffled red/blue schedule and draw order define the source transition convention. Tests retain strict deterministic physics checks and assess stochastic initialization/transition shape, support, acceptance and uniform moments; they never require equality of seeded sample realizations.
