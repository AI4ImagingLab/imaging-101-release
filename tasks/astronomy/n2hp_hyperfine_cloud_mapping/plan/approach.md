# Real-cloud hyperfine radiative-transfer inversion: implemented approach

## Unknown maps, observations and declared scope

The real IRAM 30m OphA cube supplies 501 brightness channels for a contiguous
3-by-3 sky patch. Let i index channels and y,x index sky pixels. D_i[y,x] is
background-subtracted brightness in K and nu_i is a measured channel frequency
in Hz. Four maps are inferred independently along each sightline:
p=(T_e,t0,v,sigma), excitation temperature in K, total component peak-opacity
parameter dimensionless, signed velocity km/s and positive Gaussian velocity
dispersion km/s. Sigma is standard deviation, not FWHM. Spatial pitch is
7.50000019203 arcsec and the beam is approximately 26 arcsec, so adjacent
pixels are not independent angular resolution elements. This is beam-scale
parameter mapping with no angular deconvolution, spatial regularizer or coupling.

All original float32 brightness samples are promoted exactly to float64. Use
the full 501-channel interval at each pixel, without interpolation, masking,
baseline refitting or additional spectrometer convolution. The cube selection
was fixed before local extraction to avoid a documented optically thin,
nonidentifiable corner in the larger demonstration. It does not establish
method robustness for every sightline or uniqueness of cloud parameters.

## Spectral axis and molecular calibration

Let c=299792.458 km/s, h=6.62607015e-34 J s, k=1.380649e-23 J/K,
T_b=2.73 K. The distinct radio-axis reference frequency is
nu_r=93176265000 Hz. Zero-based channel index i maps to
u_i=((i+1)-403.0960083008)*(-0.06284131109715)+2.5 km/s and
nu_i=nu_r*(1-u_i/c). Actual forward evaluation uses the supplied frequencies,
not an approximate re-created uniform axis. No periodic spectral boundary is
introduced: the analytical profiles are sampled only at these centers.

The public 15-component calibration has velocity offsets d_j in km/s and
dimensionless strengths w_j in matching order. Component reference
nu_c=93173763700 Hz is distinct from nu_r. Preserve actual W=sum(w_j)
=0.9999990000000001. Set f_j=nu_c*(1-d_j/c) and L_j=f_j*(1-d_j/c).
The second offset factor preserves the upstream finite-frequency Doppler
width/shift convention; substituting one shared nu_c or one factor is a
different numerical model. Each component has width b_j=sigma*L_j/c in Hz
and velocity shift a_j=v*L_j/c in Hz. Positive v lowers observed frequency.

## Opacity and thermal source-function derivation

For z_ij=nu_i+a_j-f_j, the opacity and brightness are

```math
\tau_i(p)=t_0\sum_{j=1}^{15}\frac{w_j}{W}
 \exp[-z_{ij}^2/(2b_j^2)],\quad
J(\nu,T)=\frac{h\nu/k}{\exp[h\nu/(kT)]-1},
\quad M_i(p)=[J(\nu_i,T_e)-J(\nu_i,T_b)](1-e^{-\tau_i}).
```

J is the Planck-equivalent brightness temperature in K; the exponent is
positive for positive T. Uniform slab source, one velocity component, common
excitation for every hyperfine component, unity beam filling and a uniform
2.73 K background define the transfer solution. The opacity is a sum of
Gaussian peak profiles, not area-normalized Gaussian densities. Thus t0 is
the sum of component peak opacities, neither maximum blended opacity nor
integrated opacity. Its integral is t0*sum_j(w_j/W)*sqrt(2*pi)*b_j in Hz.
Higher J=2-1/3-2 transitions in the full original model have tails that
underflow to zero over the bound interval; retained J=1-0 components agree
with the original forward fixtures without grouped-strength approximations.

The independent physical limits are M=0 at t0=0 or T_e=T_b. For emission
T_e>T_b, 0<=M<=[J(T_e)-J(T_b)]. Small tau gives
M approximately [J(T_e)-J(T_b)]*tau, exposing the temperature-opacity product
degeneracy. Hyperfine saturation helps separate them but cannot guarantee
identifiability. In the paper Appendix D7 a negative Planck exponent conflicts
with Appendix C2, D6 and the executed model. The positive exponent here follows
the validated source and independent thermal limits; this is an explicitly
documented typographical discrepancy, not a hidden code correction.

## Data-derived noise and objective normalization

For each y,x estimate n[y,x] as population standard deviation (ddof=0) of
observed D over inclusive radio velocity 20<=u<=28 km/s. The mean is removed
only within that standard-deviation computation, not from the fitted spectra.
Require at least two selected channels and n>0. This public-data calibration
is a per-pixel constant weighting scale. It is not a supplied full covariance
matrix or independent true noise measurement.

Residual r_i=(M_i-D_i)/n is dimensionless. Minimize chi2=sum_i r_i^2 for all
501 channels with T_e>=1e-5 K, t0>=0, unrestricted v, sigma>=1e-12 km/s.
No mask, continuum, regularizer, normalization by channel number, spatial prior
or target-informed starting value enters the optimizer. SciPy internally
minimizes half of this sum, which has the same minimizer; returned chi_square
is the full sum. The lower width bound excludes a singular expression rather
than imposing a scientifically selected minimum line width.

## Initialization, inverse steps and stopping

Initialize every sightline at (5 K,0.5,3 km/s,1 km/s). The independent original
oracle uses bounded MPFIT Levenberg-Marquardt. The local code retains the exact
physical model, objective, physical bounds and fixed initialization, but uses
the permitted generic SciPy bounded trust-region reflective backend. This
explicit optimizer substitution does not reproduce MPFIT's iteration trajectory.
There is no published custom proximal/reweighting update being replaced.

At current p the numerical Jacobian A approximates dr/dp by three-point central
differences. The local residual linearization is r(p+delta) approximately
r(p)+A delta. Generic TRF uses a variable scaling from Jacobian column norms
(`x_scale='jac'`), bounds-aware trust-region steps, and actual/predicted
decrease to adjust its trust radius. It accepts a feasible trial and repeats;
the task-specific executable code supplies M,r,bounds and initialization,
not a domain-library fit wrapper. Parameters stay in physical units.

Call least_squares(method='trf',jac='3-point',x_scale='jac',max_nfev=1000,
ftol=xtol=gtol=1e-10). SciPy's cost-reduction, parameter-step and scaled-gradient
termination criteria are optimizer-specific; they are not asserted identical
to MPFIT's identically named tolerances. Check result.success and raise
RuntimeError if unsuccessful. On success return result.x and dot(fun,fun).
The nine spectra are processed in NumPy row-major ndindex order; no warm-start
from neighboring fits is used.

For mathematical verification, define E_j=exp(-z_j^2/(2b_j^2)) and G=J(T_e)-J(T_b).
Then dE_j/dv=-E_j*z_j*L_j/(c*b_j^2),
dE_j/dsigma=E_j*z_j^2/(sigma*b_j^2), and d tau/dt0=sum(w_j/W)E_j.
For a=h*nu/k, dJ/dT=a^2*exp(a/T)/(T^2*(exp(a/T)-1)^2).
Thus dM=(1-exp(-tau))*dG+G*exp(-tau)*d tau. These explain parameter
coupling and finite-difference checks; the shipped optimizer actually uses
numerical three-point Jacobians, not an undocumented analytical derivative.

## Output maps, independent metrics and reproducibility

Output keys excitation_temperature,optical_depth,velocity,dispersion each
have shape (1,3,3), float64 (float32 also accepted), units K,1,km/s,km/s,
and axis order batch,y,x. All are finite. The standalone Judge checks positive
temperature/dispersion and nonnegative opacity before scoring. It computes
max|estimate-reference|/|reference| for each nonzero reference map, every
quantity required <=2e-4. It independently recomputes the physical model and
reports max_yx chi2/(501-4), required <=1.3. There is no map alignment,
rescaling, masking or reference-dependent physical calibration.

Independent original MPFIT repetition was identical; a stopping-sensitivity
run perturbed parameters at most 1.3421499336145399e-5 relative, fixing the
map tolerance before extraction. The original maximum reduced chi-square
was 1.2160321401642157. Reference agreement is not truth; these are independently
reproduced standard fits, not historical CLASS numbers or measured cloud truth.
Frozen fixtures verify forward opacity/brightness, noise, numerical Jacobian,
weighted loss, fitted spectra and all four maps. Independent Gaussian integral,
redshift sign, thermal limits and noiseless identifiable recovery complement
parity. Unit/CLI tests separately exercise all malformed and evaluator paths.
The active review preserves four map rows, all selected spectrum/residual groups,
four-map error chart and paired nine-pixel reduced-chi-square bars.
