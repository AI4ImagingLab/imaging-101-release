# Implemented tilted-ring mathematics

The selected acquisition is the entire author NGC2403 float32 cube, 62 velocity channels by150 by150 pixels. Its physical forward assumptions and all three output profiles are fixed in README. The baseline is an independently rerun qualified author method, not galaxy truth. This implementation uses only NumPy/SciPy primitives; no domain library or upstream source is imported.

## Observation support and units

Intensities are Jy/beam. Velocity channel k is289.610969−5.15286k km/s; x/y angular increments are−/+16.00000038756 arcsec. Fixed disk parameters are zero-based center(77,77), systemic132.8km/s, inclination60degrees, vertical sech² scale10arcsec. The model center is not the FITS reference pixel. All41 radius knots15+30j arcsec, including the outermost1215arcsec, remain required outputs.

`search_mask` estimates the median and median absolute deviation of finite nonzero observed intensities. The source float32 Gaussian conversion is sigma=MAD/0.6744888. Detections exceed median+4sigma and are grown down to median+2.5sigma. Neighborhood adjacency is Chebyshev distance1 in x/y and separation at most2 spectral channels. Qualifying high-threshold components require at least16 different sky pixels,32 voxels and two consecutive channels; the largest grown source is selected. The original connected-object implementation and the equivalent local union of connected voxels produce exactly153243 selected voxels. This is preprocessing derived only from observations, never a hidden-field mask. No output ring or residual voxel is removed by this mask.

`first_moment` sums channel intensities in source order and computes the masked weighted velocity. The numerator adds double intensity-times-velocity products then rounds each addition to float32; the denominator uses float32 additions. A zero denominator yields NaN. The native13687 supported sky pixels are retained. These missing velocity-map pixels are missing observational summaries, not missing kinematic outputs.

## Two-dimensional initialization

The source prepass fits40 rings, using the first40 supplied centers and width30arcsec. Its parameters p=(systemic velocity,circular rotation,radial expansion,PA,inclination,xcenter,ycenter) have only rotation and PA free. Initial values are(132.8,120,0,123.7,60,77,77) in km/s,degrees,pixels. Positions projected into the disk define azimuth theta, and the line-of-sight model is

```math
f(x,y;p)=p_0+(p_1\cos\theta+p_2\sin\theta)\sin i.
```

Here i is inclination; phi is PA. Set X=−(x−x0)sin(phi)+(y−y0)cos(phi), Y=[−(x−x0)cos(phi)−(y−y0)sin(phi)]/cos(i), r=sqrt(X²+Y²), cos(theta)=X/r and sin(theta)=Y/r. Angles in trigonometric functions are radians, while p stores degrees. Samples must lie strictly between each annulus's boundaries and outside15degrees of the projected minor axis. Weights are cos²(theta). Traversal is x then y. Geometry and accumulated outer residual q are float32.

`prepass_derivatives` evaluates the source analytic rotation and PA derivatives. With F=pi/180, c=cos(theta), s=sin(theta), and radial expansion zero, these are sin(i)c and F V(1−sin²(i)s²)s sin(i)/cos(i). Analytic derivatives are additionally checked against finite differences; agreement with upstream matrices alone is not a physics check.

`prepass_normal_equations` computes the weighted squared residual chi, J-transpose-W-J matrix A and J-transpose-W-residual vector b in float64 using source float32 model/derivative assignments. `_damped_prepass_fit` scales A by the square roots of its diagonal, replaces its normalized diagonal by1+lambda, performs explicit Gauss-Jordan inversion and computes the parameter increment. Lambda starts at0.01, is divided by10 before an attempt, and is multiplied by10 until the residual improves or exceeds1e10. Relative squared-residual tolerance is0.001; the source inner budget is100 iterations. All updated parameters round to float32.

`initialize_rings` repeats that fit while reselecting the projected annular data. If a new field residual fails to improve, it tries alternating flip factors1,−1,0.5,−0.5,... times the fitted difference. The first50 outer iterations use relative residual change0.001; subsequently the stop test is that all scaled parameter increments are below0.1 in their respective stored units. It retains the source100-iteration maximum.

The original code copied41 entries from40-element prepass arrays. ASAN proved an out-of-bounds read. The qualified oracle and local extraction copy only valid entries1..39, retain the user initialization on ring40, retain rotation120 on ring0, and set PA0=PA1. This is a disclosed memory-safety correction, not an unchanged upstream run. Every requested output ring remains present.

## Cloud acquisition operator

`internal_rings` converts velocities to m/s, radii and heights to arcmin, and angles to radians with source float32 container assignments. It linearly interpolates user rings onto radial quadrature spacing0.75 angular pixel=0.2000000048445arcmin. A fitted annulus has two radial edges and constant physical parameters. Its newly constructed source ring container has radsep0; its bounding extent is ceil((outer edge+3z0)/pixel), clipped to the native150² field. This convention differs from erroneously adding another half-width to an already expanded edge.

For each internal radius r and quadrature spacing dr, its area in pixels is a=2pi r dr/pixel_area. With source cloud parameter CDENS10 and density rho, the number of clouds is round(10 rho a). If round(n/a)<10, the source instead uses round(10a). Round is halfway-away-from-zero. The density is arbitrary1 during annular kinematic fitting because flux is normalized against observations. CDENS is an integration parameter, not a measured gas density.

Each cloud draws U1,U2,U3 uniformly in[−1,1), yielding R=sqrt((r−dr/2)²+2dr r|U1|), azimuth=2pi|U2| and height=atanh(U3)z0. This yields uniform annular area and a normalized sech² vertical law. Sky projection and LOS velocity are those defined in README. Instrumental sigma is the source float32 one-channel FWHM-to-sigma coefficient multiplied by channel width. Its variance is added to the squared intrinsic dispersion; the commented source channel-box variance subtraction is not active and is not silently introduced here. Spatial and spectral indices use nearest-integer assignment with halfway cases away from zero. Off-box clouds consume spatial draws but no velocity draws.

`CloudRandom` implements the MT19937 raw-word convention using general NumPy uint32 generation, then explicitly evaluates float canonical uniform values and the Marsaglia polar normal transformation with float32 assignments. Every forward evaluation resets the selected integration stream, matching source common-random quadrature. Scientific quality tolerances come from complete independent source seeds−1,−2,−3, not same-seed agreement. Fixed-geometry samples may be memoized by `_cloud_samples`; fitted profiles, losses and forward cubes are never cached. The cache is mathematically independent of rotation/dispersion and changes neither sampling nor the inverse updates.

There are62 velocity draws per cloud for the selected positive dispersions. Each cloud contributes rho*a*pixel_area/(n*62), then the source density-as-flux conversion3600/channel_width_m_s*1000. Repeated voxel additions remain float32. The selected profile units therefore produce Jy/beam after the observation-derived normalization; no independent hydrogen density is inferred.

`beam_kernel` explicitly constructs the elliptical Gaussian beam. Source `Convpars` stores twice the beam orientation in float32 before halving it; that convention is preserved. Gaussian values below1e−4 of its peak are zero, extents are source-rounded, and the retained kernel is normalized to sum1. `beam_convolve` uses zero-padded linear SAME convolution independently by channel, suppresses values below1e−12 and rounds to float32. These stages have independently captured original arrays and analytic positivity/normalization/boundary checks.

## Objective and custom inverse update

The source scales the entire cube to maximum10 before each fitting stage, and restores it afterward with float32 assignments. The noise estimate is scaled/restored likewise. This round-trip is preserved between stages. Channel noise weights remain1. Within the projected annulus, descending y,x,channel float32 sums determine the gain mu=(sum masked observations)/(sum model). A zero model sum gives gain0. The annulus is selected inclusively at its inner and outer edges; its azimuthal weight is cos²(theta).

The objective is the README weighted absolute residual and blank penalty. Positive observations retain their actual values; only nonpositive observations are replaced by the noise standard deviation. A mask-false voxel with zero model is skipped; a mask-false nonzero model increments the blank count. The blank fraction is evaluated in float32; weighted residuals are accumulated in float64. `azimuthal_loss` returns the normalized model and the independently testable summary: observed sum,model sum,gain,weighted absolute residual,count,blank count,number of spatial pixels,objective.

`nelder_mead` explicitly implements the original downhill simplex, rather than substituting SciPy's different conventions. It starts at the supplied point, adding one axis step to each extra vertex. Rotation step is30km/s above40km/s, otherwise5. Dispersion step is20 above80,10 above20,0.2 at or below2, otherwise5km/s. PA step is10degrees. The highest/second-highest/best selection and last-tie convention match the source. Reflection factor is−1, expansion2, contraction0.5 and shrink0.5. Trial arithmetic is double before float32 assignment; simplex sums and stored objective values are float32. The relative high-low objective range threshold is0.001, with denominator epsilon1e−10 and source evaluation budget200 times dimension, excluding initial vertices. Exceeding the budget raises for the complete task; it never silently drops a difficult ring.

`ring_objective` applies the original trial-repair rules. Rotation stays within its initial±1000km/s interval and the original global limits, whose lower endpoint is at least0. Out-of-range trials are repaired with the source's persistent uniform stream. Dispersion has limits0.1..1000km/s and source uniform repairs around those endpoints. PA is clipped to initial±15degrees and original global PA bounds. The trial array is mutated before simplex replacement, as in the source. Only source-supported fitted spectral rotation speeds are accepted.

`reconstruct` fits all41 annuli in rotation,dispersion,PA. It computes PA median/MAD and follows the actual source low-scatter auto branch: scatter MAD/0.6744888<=3degrees selects a constant median PA. That condition is verified for each independently frozen source run. The binding does not implement or silently substitute the unselected high-scatter Bezier branch; an unexpected branch raises. The second complete fit frees rotation/dispersion and fixes regularized PA. All fields, losses, counts and prepass diagnostics are returned.

## Full-cube observation consistency

`zero_moment` computes the masked integrated line intensity in Jy/beam km/s, preserving original float32 accumulation. `azimuthal_profile` uses the author's2x2 subpixel annular integration, counts both valid and missing image summaries, and estimates face-on nuisance surface brightness with cos(inclination). It returns the integrated sum,valid/blank subpixel counts,sky brightness and face-on brightness. Missing summaries are not treated as measured zero brightness in this estimate.

`full_model` scales the lowest positive profile into[0.1,10] by powers of ten, interpolates that brightness and the fitted kinematic profiles to the full disk, and applies the complete cloud/beam operator. The first radial knot is moved from15 to0arcsec and the last from1215 to1230arcsec as in the source final-model output. The full150² by62 cube is returned in Jy/beam; no evaluation mask or fitted observation gain is used to reduce its residual. Forward RMS is sqrt(mean((model−observed)²)) over all1395000 native voxels.

## Evaluation and interpretation

All three output profiles are evaluated; separate rotation/dispersion range-NRMSE avoids mixing units or hiding a difficult component. PA uses RMS of degree differences wrapped to[−180,180). There is no amplitude,constant-offset or registration fitting. Actual truth, if present in an evaluator installation, takes priority over a method reference and malformed truth never falls back to the method result. Baseline references are explicitly method outputs, not measured truth.

Source-only quality thresholds were frozen before local extraction: rotation range-NRMSE0.017034759657074735,dispersion0.10383335124662954,PA wrapped RMS0.3988151550292969degrees,full-cube RMS0.005076774190082738Jy/beam. Field limits are1.5 times the worst independently observed complete-source seed discrepancy; forward limit is1.1 times the worst complete-source residual. Uniform,zero and shuffled fields fail nonvacuously. These limits express reproducibility of a conditional stochastic method, not physical calibration or posterior confidence intervals.
