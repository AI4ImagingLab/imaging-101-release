# Implemented real-profile scalar-wave migration

## Scope, units and acquisition

The measured GSSI dune line is512 recorded-time samples ×1040 traces, with no spatial subsetting. Original16-bit unsigned values are converted by the upstream reader to signed counts by subtracting32768, preserved exactly as float64. Time axis is linspace(0,48,512)ns; its stored first interval is0.09393346379647749ns. Header trace density50/m gives x=arange(1040)/50 metres, not an endpoint-stretched line. The independent36-station surface survey is available for display. Nominal electromagnetic velocity v=1.4e8m/s (.14m/ns), delay t0=5ns. Neither velocity nor gain is a recovered material property.

Unknown I(x,z) is a signed migrated initial wavefield in instrument amplitude units, not absolute permittivity or true source strength. The scalar exploding-reflector wave equation is Ptt=(v/2)²(Pxx+Pzz); observed background-free data are the surface wavefield P(x,0,t), and image is P(x,z,0). The half-speed models two-way propagation. Positive z is downward, output dz=v*dt_seconds/2=0.006575342465753425m, xmax20.78m, zmax3.0049315068493145m. Finite-aperture planar homogeneous approximation does not model the surveyed slope in wave propagation. Terrain display shifts columns afterward and is explicitly not a topography-aware migration.

## Time correction and backgrounds

Use continuous linear interpolation at target recorded times t0+j*dt, j=0..457; dt is the stored raw interval, and n=floor((48-t0)/dt)+1=458. This preserves the nominal5ns calibration without nearest-index rounding. Last sampled recorded time is47.92759295499022ns. Compute adjacent input indices with searchsorted and interpolate linearly between the two recorded samples. All1040 traces remain present. Original GPRPy's nearest-index setZeroTime differs; the selected task uses this explicitly declared continuous-time preprocessing before the unmodified ImpDAR kernel and does not claim unchanged GPRPy demonstration figures.

Subtract the mean over458 time samples independently for each trace (constant-offset removal), then subtract the mean over1040 traces independently at every corrected time (common mean-trace removal). These operations match the original GPRPy global-window dewow/remMeanTrace formulae on the same interpolated observations. No amplitude division, gain compensation, trainable parameters, frequency-band fitting or noise-free measurement is supplied.

## Fourier aperture and dispersion

At row j and trace i multiply by W=hv, h=min(1,i/100,(1039-i)/100), v=min(1,j/40,(457-j)/40). This separable linear taper approaches zero at each image boundary before periodic extension; widths are fixed convention, not selected by hidden answers. Input remains float64.

Use np.fft.rfft2(weighted,axes=(1,0)): the full horizontal FFT precedes the real temporal FFT, producing FK shape(230,1040), complex128. Temporal omega=2π*rfftfreq(458,dt_seconds), horizontal kx=2π*fftfreq(1040,.02), vertical kz=2omega/v. Each query frequency is omega*=v/2*sqrt(kz²+kx²). Frequencies are rad/s; kx,kz rad/m. Source and local code retain forward FFT unnormalized and inverse normalized by458*1040.

The output spectrum KK(kz,kx)=C*FK(kx,omega*), C=kz/sqrt(kx²+kz²). The continuous transform Jacobian contains a v/2 factor, but this cancels the changed vertical frequency-cell spacing when storing the sampled output at z=v*t/2. Thus the discrete executable obliquity is dimensionless, matching upstream; do not multiply by v/2 again.

## Sampled remap and inverse

ImpDAR creates linear RectBivariateSpline objects in kx/omega. Every query uses an existing horizontal grid point; consequently its bilinear interpolation reduces exactly to1-D linear interpolation between adjacent temporal bins in each horizontal column. Local code evaluates that formula explicitly. searchsorted gives lower bin l clipped to0..228; weight a=clip((omega*-omega_l)/(omega_(l+1)-omega_l),0,1); interpolated value=(1-a)FK_l+aFK_(l+1). This preserves the original spline's endpoint extension beyond the maximum measured omega.

Only output rows0..228 are populated; final temporal Nyquist row229 is zero as in the original loop range(snum//2). Multiply by C and set KK[0,0]=0 to remove undefined DC. Other kz=0 coefficients are zero because C=0. Inverse np.fft.irfft2(KK,axes=(1,0)) gives a real458×1040 image. Odd temporal sizes are rejected explicitly because the upstream implicit inverse length would otherwise drop a sample. The bound458-point corrected grid is even. This is a direct closed-form spectral inverse; no initialization, optimization objective, iteration, step size, line search or stopping rule exists.

remap_adjoint distributes each complex output coefficient to the same two temporal input bins with weights1-a,a; final output row contributes nothing. It is the exact discrete Euclidean adjoint of remap and is tested by a complex dot-product identity. It is a diagnostic for interpolation mathematics, not a replacement reconstruction method.

## Quantified spectral boundary approximation

Endpoint extension is a numerical approximation, not a strict bandlimited physical response. For this actual data, queries above Nyquist occupy2.0656354515050167% of spectral grid points, but their migrated energy fraction is4.983924646640737e-7; the last measured temporal-bin energy fraction is9.784685199529186e-8. A separately computed bandlimited diagnostic instead sets only omega*>omega_max coefficients to zero. Difference from the selected upstream convention: relative image L2=0.0007059691669358328, dynamic-range NRMSE=2.9671960731516316e-5, maximum absolute difference23.18485085132905counts. This small measurable effect is preserved and displayed, not hidden or presented as an exact bandlimited wave operator. The standard baseline/thresholds continue to evaluate the upstream convention, not this optional diagnostic.

## Physics validation and scientific comparisons

An independent kinematic point-diffraction helper evaluates t=2sqrt((x-xs)²+depth²)/v with a Ricker pulse and cylindrical1/sqrt(R) spreading. It generates only a small analytic test, not the bound observations. Its migration must focus at the known position/depth within spatial/temporal sampling. Additional tests verify the scalar dispersion relation, zero-horizontal-frequency sinusoid identity (away from removed DC/Nyquist), complex interpolation-adjoint identity, real linearity, DC/common-background removal and zero response. Full-source fixtures separately validate raw/preprocessing/FK/KK/scaling/images for .12,.14,.16m/ns.

Notebook actively reconstructs all3velocity hypotheses plus the zero-outside-band diagnostic. Output remains the selected .14m/ns source-convention image. Compare the upstream result, full fresh image, numerical difference, physical boundary diagnostic, velocity sensitivity and terrain display. Different velocity images share the explicitly labelled nominal .14m/ns depth grid in the comparison panel, so focusing changes are not hidden by axis rescaling.

For terrain display, interpolate36 surveyed heights to trace positions with PCHIP. Compute shifts=round((surface-min(surface))/dz); offset=max(shifts)-shifts; place each migrated trace at that row offset in a NaN-padded display image. This exactly reproduces original GPRPy correctTopo at the same coordinates. It is not a required output and its undefined exterior is deliberately masked, not zero-filled data.

## Evaluation

Actual subsurface truth is unavailable. Independent upstream original migration on the exact bound observation supplies baseline_reference. NCC is cosine similarity of all flattened signed samples without centring; NRMSE is RMS error divided by reference dynamic range. No alignment, crop, scaling, magnitude, output-file guessing or terrain transformation. Fixed nonzero boundaries NCC≥.99999999 and NRMSE≤1e-7 were frozen before local implementation. They measure numerical method reproducibility, not absolute material accuracy; independent physical checks and the documented small numerical boundary approximation are separate gates.
