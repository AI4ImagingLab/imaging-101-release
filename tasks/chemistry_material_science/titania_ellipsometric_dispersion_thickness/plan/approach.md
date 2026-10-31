# Implemented ellipsometric inverse

## Selected observations and conditional model

The benchmark retains all926 points of the exact author-defined400–800nm transparent-band example from a real Sentech spectrum of400cycle ALD TiO2 onSiO2/Si. The unmodified original file contains1227 points319.84071–850.03158nm. It is archived whole, not called the public full analysis spectrum. The source chose the window before this task: transparent Cauchy physics and Aspnes silicon calibration motivate it. No postfit point removal occurs. This is a nonspatial optical dispersion/thickness inverse, not a voxel image.

The source model angle is70deg, although the recorded instrument angle is70.06deg. Preserve70 in reconstruction and Judge; evaluate70.06 only as explicitly labeled held-parameter sensitivity. Known oxide thickness276.36nm and nOx=1.452+3600/L² are the author's conditional model constants from an asserted separate substrate fit. Its corresponding original observation is unavailable. The currently bundled Basicusage NeXus instead describes a different2nm oxide specimen and freshly fits2.065789nm; never identify it as the276.36nm calibration. The author supplies this fixed conditional substrate; no evidence of target-derived origin was found, while the matched original calibration data and acquisition chronology remain unavailable.

## Unknowns, units and forward physics

The parameter vector p=(n0,a,b,d) has four entries. For L equal to wavelength expressed numerically in nm, n(L)=n0+100a/L²+10^7b/L⁴, with k=0. Here a and b are source-scaled numerical coordinates, not conventional B,C: B=100a nm², C=10^7b nm⁴. d is physical model thickness in nm. n0 and n(L) are dimensionless. A mixed-unit dot product/NCC on p has no evaluation role.

Layer indices are N=(1,nTiO2,nOx,nSi), with nSi=n+i*k a supplied independent calibration. The source material API squares index to permittivity then the2x2solver takes its principal complex root; local code preserves that order, including trial values. Complex Snell angles satisfy N0*sin(theta0)=Nj*sin(thetaj). theta0=70*pi/180. At the incident and substrate boundaries choose forward angles by Im(N*cos(theta))>0 when its magnitude exceeds1e-10, otherwise Re(N*cos(theta))>0; replace theta by pi-theta if needed. Finite film angles keep the source principal complex arcsine. The calibrated passive stack is nondegenerate.

For incident/transmitted indices Ni,Nt and cosines ci,ct, the four explicit Fresnel amplitudes are

```math
r_s=(N_i c_i-N_t c_t)/(N_i c_i+N_t c_t),\quad
r_p=(N_t c_i-N_i c_t)/(N_t c_i+N_i c_t),
t_s=2N_i c_i/(N_i c_i+N_t c_t),\quad
t_p=2N_i c_i/(N_t c_i+N_i c_t).
```

For wavelength L, normal wavevector kz(j)=2*pi*Nj*cos(thetaj)/L and film phase beta(j)=kz(j)*Dj, where D1=d,D2=276.36nm. The interface matrix at the first boundary for each polarization is I=(1/t)*[[1,r],[r,1]]. Append each film and its following interface by multiplication with (1/t)*[[exp(-i*beta),r*exp(-i*beta)],[r*exp(i*beta),exp(i*beta)]]. Matrices multiply from air toward substrate. The final reflected amplitude is M10/M00; source scalar summation/grouping is preserved using a vectorized2x2einsum. Returned complex rho=Rp/Rs.

Measured rhoObs=tan(Psi*pi/180)*exp(-i*Delta*pi/180). This negative Delta exponent, n+i*k convention and passive branch are explicit and tested. The wavelength arrays and resulting spectrum remain in their complete original increasing order. No interpolation of observations, amplitude renormalization or phase unwrap is performed.

## Objective, finite bounds and inverse updates

The1852component residual concatenates all926 Re(rhoObs-rhoModel), then all926 Im(rhoObs-rhoModel). Minimize its unnormalized Euclidean sum of squares. There is no separate Psi/Delta weighting, robust loss, prior or spectral smoothing. Every call evaluates the complete TMM at every point.

The native finite bounds are min=(-100,-40000,-40000,0) and max=(100,40000,40000,40000). Initial p=(2.236,451,251,20). Preserve lmfit's two-sided Minuit-style transforms explicitly:

```math
u_j=\arcsin\left(2\frac{p_j-l_j}{h_j-l_j}-1\right),\qquad
p_j=l_j+(\sin u_j+1)(h_j-l_j)/2.
```

Here l,h are lower/upper bounds and u unconstrained internal coordinates. Scalar operation ordering matches the source. The forward mathematics is local; no lmfit or pyElli package is imported. SciPy's general MINPACK lmdif primitive, exposed through scipy.optimize.leastsq, provides finite-difference Levenberg–Marquardt updates in u. Its source parameters are ftol=xtol=1.5e-8, gtol=0, epsfcn=1e-10, factor=100, no analyticDfun or diag. The four-variable source default limits its adjusted nfev counter to10000. Its counter starts at-2 before MINPACK for setup/check probes, so10002 raw user-residual callbacks are allowed; callback10003 errors before evaluating the residual. The local explicit raw-call guard preserves this accounting, with MINPACK maxfev=20000. Successful statuses1–4 are accepted; other statuses error instead of emitting a substitute. No boundedTRF substitution or stopping-threshold change occurs.

Diagnostic evaluation histories include finite-difference perturbations and repeated checks and are not optimizer iterations or accepted-step trajectories. Source and default local runs report32MINPACK evaluations. Raw local call counts can include scipy setup probes. External covariance, when available, is transformed by diag((h-l)*cos(u)/2) on both sides and multiplied by residualSSE/(1852-4), matching native conditional covariance scaling. It is not an independently calibrated physical uncertainty estimate.

## Identifiability, comparisons and evaluation

All four parameters and the full926index field are required. Fixed reference is a validated source method result, not actual film truth. Before local extraction the ordinary source2x2run was repeated from process restart with bitwise-equal arrays. A native4x4alternative fit with the same data/initialization/bounds differs by approximately(9.1e-11,4.5e-7,5.8e-7,7.8e-11). This supports the prelocal nonzero parameter tolerances(1e-7,5e-4,1e-3,1e-7), source-scaled coordinates and nm respectively. nfield dynamic-rangeNRMSE<=1e-7 and parameter-to-field consistency maxabs<=1e-10 also apply. Complex-rho forward relativeRMS<=0.02 was frozen before any source fit; measured original is0.012221. Every gate is conjunctive. No rescaling, crop, reference alignment or hidden parameter fit is part of evaluation.

Independent source-stage tests use separate recursive reflection, single-interface/zero-layer/Brewster/energy limits, all-grid parity and scaledJacobian rank. The full926complex response has four local independent directions; condition is about2713 for diag(max(abs(p),1)) scaling. Coefficients remain correlated: source n2conditional relative standard error is about13.6percent. Numerical repeatability gates must not be described as this material's true precision.

Notebook comparisons retain observations, initial actual forward model, original2x2fit, independent original4x4fit and fresh full localfit; actual additional initializations; evaluation-call histories; n and all4parameters; conditional covariance/correlation; held70/70.06degree response; and passive s/preflectance. Additional fits/plots are freshly computed or explicitly labeled archived source measurements, never fabricated alternatives. No training or synthetic benchmark data generation occurs.
