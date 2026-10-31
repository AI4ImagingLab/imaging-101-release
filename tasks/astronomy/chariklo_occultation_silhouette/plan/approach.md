# Implemented Monte Carlo silhouette inversion

The physical unknown consists of five projected elliptical-limb parameters, not a textured surface or a full3Dshape. Parameter order is center_f[km],center_g[km],semi-major radius[km],oblateness[unitless],pole position angle[degrees modulo180]. Public contour angles are360equally spaced values0inclusive to2pi exclusive. Their coordinates are a redundant representation of these same five unknowns.

## Geometry and observations

The public f/g positions at ten observed contact times are generated from known star, observer, and body ephemeris calibration by the original SORA Chord.get_fg implementation. Paper section2.3defines the orthographic tangent-plane positions: observer minus ephemeris-body coordinates, with east/north axes. Original source uses topocentric calibrated star geometry and bundled SPICE kernels. Gaia-style astrometry is explicitly supplied from the author's displayed values, avoiding a mutable online catalog query. The bundled kernel's actual target ID20010199 replaces the stale example ID2010199. This selects the bundled object's identity rather than changing trajectory mathematics. The reference center is explicitly geocentric, avoiding online MPC geocenter metadata. Earth orientation uses Astropy's bundled offline data. These adaptations and coordinate-origin changes are recorded in validation. Neither the hidden shape nor reconstructed contacts determine this calibration.

For measured point xi=(fi,gi), velocity vi and timing weight sigma_ti, use scalar distance sigma_i=norm(vi)*sigma_ti. There are five positive instrument records from four distinct sites; the two Windhoek instruments provide correlated geometric coverage, not independent spatial chords. The author's timing errors are intentionally enlarged for teaching visualization, and are not claimed as original published uncertainties. The Hakos constant-one example flux is used for declared non-occultation status and cadence only, not as raw photometry. Standard benchmark negative samples span the actual loaded cadence file's first/last time at0.5s steps. This deliberately imposes a continuous-path constraint; the exposure-only alternative is retained as a separate review comparison.

## Ellipse and objective

Let a be semi-major radius, b=a(1-o), o oblateness, phi pole angle, and center(f0,g0). At polar angle theta around the center,

r(theta)=ab/sqrt([a sin(theta+phi*pi/180)]²+[b cos(theta+phi*pi/180)]²).

Coordinates are center+r(cos(theta),sin(theta)). For each measured point compute theta_i=atan2(gi-g0,fi-f0), project radially to the ellipse, and accumulate chi²=sum_i[(fi-fmodel_i)²+(gi-gmodel_i)²]/sigma_i². This is the source radial uncertainty convention. No geometric orthogonal-distance substitute or quartic least-squares wrapper is used. No regularizer is added; finite search ranges and the elliptical model are explicit priors.

## Exact original search

Use legacy MT19937 seed24713 and batches100000. In each batch draw all center_f, then all center_g, then all radius, then all oblateness, then all angle values. Draw uniformly from center[5.416,-3.747,138,.093,90] plus/minus widths[10,20,50,.20,90]. Clip oblateness to[0,1] exactly as source, so a boundary mass at zero is preserved. The center ranges were translated from the author's displayed old-kernel reference origin to the current bundled-kernel origin; widths and search semantics are unchanged. No result-dependent centering adjustment occurs during optimization.

Evaluate every candidate against all ten contacts. Retain candidates with chi² strictly below the minimum of that particular batch plus10. The source uses the batch minimum, not the running global minimum. Append batches until at least20000candidates have been retained. On the bound observations this yields20621. There is no gradient, step size, line search, convergence update or synthetic iteration. Stop only on the accepted-count rule. Alternative seeds are explicitly labeled sensitivity experiments, not replacement benchmarks.

## Negative chord and output

For every retained candidate evaluate radial clearance distance(path,center)-r(theta(path)). As in original filter_negative_chord, evaluate samples within1.5a of the candidate center; farther points are necessarily outside. Keep only candidates whose every evaluated clearance is strictly positive. The source continuous-step filter keeps19776of20621candidates. Select the global minimum-chi² candidate among these. It gives f0=10.761326206984421km,g0=-3.106247004480702km,a=137.7901561278855km,o=.07219177919178044,phi=120.04331554578464degrees. These are independently reproduced source measurements, not truth or hard-coded solver results.

The original author notebook also compares exposure-aware filtering. Its unmodified function is separately executed using the author's1s exposure/cadence file and0.1s internal sampling; it retains19975candidates and the same best model. A separate precomputed exposure-geometry comparison allows the notebook to repeat this filtering with the cleaned ellipse operator without an upstream import. The mandatory benchmark still uses the full continuous path, as specified publicly. Positive-only, continuous-negative and exposure-negative ensembles are individually identified.

## Evaluation and nonuniqueness

Every parameter is checked separately. Angle differences wrap modulo180, rather than raw subtraction. Boundary samples use cosine NCC and RMS/reference dynamic range with no fitted translation, rotation or scale. Nonzero parameter/contour tolerances are fixed before cleaned scoring from the independently reproduced negative-filtered source ensemble satisfying chi²<minimum+1:1.1times maximum departures from the source best, with analogous NCC departure. These are method-ensemble uncertainty tolerances, not proof of unique physical parameters.

The standalone Judge independently constructs ellipse quadratic form Q=R^T diag(1/a²,1/b²)R, where R rotates by phi. For each observed local ray x+v*t, solve (v^TQv)t²+2(x-center)^TQv*t+(x-center)^TQ(x-center)-1=0. Choose the contact time shift nearest zero and compute RMS(t/sigma_t) over all ten. Its bound is1.1times the independently reproduced source result. Missed trajectories separately require count0; for their finite time statistic use the quadratic vertex, without claiming it is a contact. The negative path independently requires no positive radial penetration. Submitted contour coordinates must agree with the submitted parameter geometry to1e-8km. All required finite metrics must pass. Actual truth has priority if later supplied; otherwise the original method result is a baseline. Sparse geometry, example timing weights, unmodeled topography and shared-station correlations remain limitations.
