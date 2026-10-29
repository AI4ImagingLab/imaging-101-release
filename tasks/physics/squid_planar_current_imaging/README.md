# SQUID Planar Current Imaging

> Reconstruct a spatial two-component planar current field from a scanning SQUID flux image despite sensor blur, height attenuation, incomplete field of view and unknown absolute calibration.
> Domain: physics | Keywords: magnetic imaging, current tomography, spectral regularization | Difficulty: Hard

## Background

Electric currents generate magnetic fields that can be detected without contacting the conducting region. A superconducting quantum interference device converts very small magnetic flux changes into an electrical signal. Scanning the sensor over a surface produces an indirect image of current flow through the field that it creates above the sample.

Magnetic current imaging supports inspection of electronic devices and spatial studies of conductive materials. The sensor averages over a finite area, and increasing its height suppresses fine spatial information. Measurements cover only a finite region and include noise, making recovery of detailed current patterns unstable. A planar, quasistatic model supplies a useful constraint when the conducting layer is thin compared with relevant spatial scales.

## Problem Description

The input is a real scanning SQUID lock-in flux response obtained while driving an approximately600Hz,1microampere AC current. Recover both components of a relative planar current field. Absolute scale and direction are not calibrated: the independent dipole used to measure the sensor response has unknown magnetic moment and the deposited lock-in orientation is not identified with an absolute right-hand +Bz direction. The requested signed field follows the **given author response convention**, without a fitted global sign or gain.

For the discrete model, use the negative-exponent forward Fourier transform, F[f](kx,ky)=sum f(x,y)exp[−i(kx*x+ky*y)], with inverse normalized by the number of pixels. For nonzero radial frequency K, the effective deposited response convention is


```math
\widehat f(k_x,k_y)=\frac{\mu_0}{2i}\,e^{-Kh}P(k_x,k_y)
\frac{k_x\widehat J_y-k_y\widehat J_x}{K},\qquad
k_x\widehat J_x+k_y\widehat J_y=0,\qquad K=\sqrt{k_x^2+k_y^2}.
```


Here f is the measured signed flux response in its deposited numerical units, Jx and Jy the unknown relative sheet-current components, kx and ky angular spatial frequencies in rad/um, h the sensor height in um, P the discrete sensor transfer, i the imaginary unit, and mu0 the numerical vacuum-permeability coefficient specified below. Relative-current units absorb the unknown moment, flux-unit conversion and micrometer-to-meter scale; these numbers must not be interpreted as A/m. Under this negative-exponent transform, the stated signed response is the negative of the standard right-hand +Bz Biot-Savart transfer before calibration; preserve the stated equation instead of assuming the unverified absolute lock-in polarity. The real calibration values and their signs are supplied, with no sum normalization.

The raw finite image is extended by100columns on each side and75rows on each side with `linear_ramp` padding to zero endpoint values, then treated as periodic on the full301×401grid. This is a fixed boundary assumption, not additional measurements. The sensor transfer is the centered coarse calibration shifted to index-zero origin with ifftshift and transformed by the same FFT. Its array center(row150,column200) is the discrete kernel origin, irrespective of the small measured-coordinate center offset. The existing coarse kernel retains its original index association; its source physical pitch differs slightly from the magnetic-image pitch, as documented below. No new physical resampling or fitted center shift is allowed.

Uniform current is invisible at K=0; set the mean of **both** components to zero. Fine spatial frequencies are unstable. The requested identifiable, regularized field uses the fixed radial Hanning response H(K)=0.5*(1+cos(pi*K/Kc)) for K≤Kc and zero otherwise, with Kc=sqrt(2)rad/um; apply H to the padded observation spectrum and discard its DC component for the forward consistency check. This filtering defines the requested effective spatial resolution, not a new observation.

Save `output/reconstruction.npz` containing:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 2, 301, 401) | float64 | Real signed relative current, a.u.; axes batch,component,padded-y,padded-x; component0=Jx and component1=Jy |

Rows and columns increase with physical y and x. Full output coordinates extend the measured coordinate axes by the stated pads while retaining their pitch: x from−258.48to258.48um, y from−207.35to135.85um. The native observed window is rows75:226,columns100:301, but **the full periodic field is required** so the forward operator closes without truncating edge currents. Values outside that window are boundary-dependent inferred continuation, not known physical current. No crop, component swap, fitted translation/rotation, magnitude conversion, fitted sign or scaling is applied in evaluation. Both signed components are compared separately using cosine similarity and relative L2 error; filtered forward consistency, spectral divergence and zero-mean gauge must also hold. Reference-method agreement does not establish actual current truth.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| flux | (1, 151, 201) | float64 | Real measured AC lock-in flux response, deposited Phi0/A units; axes batch,y,x |
| xs | (1, 201) | float64 | Measured magnetic-image x coordinates, um |
| ys | (1, 151) | float64 | Measured magnetic-image y coordinates, um |
| psf_native | (1, 209, 209) | float64 | Independent out-of-plane dipole calibration, deposited mPhi0 units; axes batch,y,x |
| psf_xs | (1, 209) | float64 | Native calibration x coordinates, um |
| psf_ys | (1, 209) | float64 | Native calibration y coordinates, um |
| psf_coarse | (1, 301, 401) | float64 | Existing zero-padded/cubic-spline calibration prepared on the original coarse index grid, mPhi0; use these signed values directly for P |
| psf_coarse_xs | (1, 401) | float64 | Actual original coarse-calibration x coordinates, um; documents source sampling approximation |
| psf_coarse_ys | (1, 301) | float64 | Actual original coarse-calibration y coordinates, um; documents source sampling approximation |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation type | measured scanning SQUID AC lock-in flux | Phi0/A | Real deposited response; Phi0 is the flux quantum |
| Drive | approximately600Hz,1microampere AC | Hz,microampere | Context only; unknown dipole moment prevents absolute current conversion |
| Sensor/source standoff | 1.0 | um | Known height in the selected transfer |
| Measured shape | 151rows,201columns | pixel | Full measured acquisition retained |
| Magnetic-image pitch | dx=1.2924,dy=1.144 | um | Row y,column x; both increasing |
| Measured x interval | [−129.24,129.24] | um | Deposited spatial origin |
| Measured y interval | [−121.55,50.05] | um | Deposited spatial origin |
| Native PSF pitch | dx=0.0718,dy=0.0715 | um | Independent209×209dipole scan |
| Native PSF center coordinate | x=−0.0234,y=−0.053 | um | Center array sample; no target-fitted registration |
| Native PSF interval | x[−7.4906,7.4438],y[−7.489,7.383] | um | Actual native coordinates supplied |
| Coarse PSF pitch | dx=1.295272,dy=1.1473366666666323 | um | Original physical sampling differs from magnetic pitch by about0.2222% and0.2917%; effective index kernel preserves this approximation |
| Coarse PSF center | row150,column200; x=−0.0234,y=−0.053 | pixel,um | ifftshift sets array center to zero displacement; measured center offsets are not fitted away |
| Coarse calibration construction | symmetric zero-padding then bicubic rectangular spline | dimensionless | Prepared independently from the native dipole calibration; actual result supplied to avoid huge fine-grid work arrays |
| Boundary extension | x_pad100,y_pad75,linear_ramp end_values0 | pixel,flux unit | Model assumption on flux, followed by periodic discrete operator |
| Periodic grid | 301rows,401columns | pixel | Full current field and full forward support |
| Permeability coefficient | 1.25663706127e−6 | SI numerical value | Original SciPy vacuum coefficient; relative-current scale absorbs unit conversions |
| Fourier convention | negative exponent,unscaled forward,1/(301*401)inverse | dimensionless | Angular frequencies exactly2*pi*fftfreq, shifted for display/computation |
| Current DC | zero in both components | a.u. | Explicit gauge for unobservable uniform sheet current |
| Absolute calibration | unknown dipole moment and absolute lock-in polarity | dimensionless | Signed relative field follows stated effective transfer; no A/m or absolute-direction claim |
| Noise | existing measured realization; variance unspecified | flux unit | No invented Gaussian variance or target-derived denoising input |

All public calibration was obtained independently of the current image. Native calibration is retained for traceability; the supplied coarse kernel is the operative fixed discrete calibration, and does not imply exact equality of its physical sampling with the flux grid.

## Method Hints

Fourier Biot-Savart inversion combines the planar divergence-free constraint with the out-of-plane response to recover the transverse current components. Apply a radial Hanning low-pass window before dividing by the height/sensor response; otherwise weak high-frequency signals can be greatly amplified. Handle the zero-frequency nullspace explicitly and check Hermitian symmetry before retaining a real current field. A full-field forward check and a divergence check provide complementary validation of the two components.

## References

Roth, Sepulveda and Wikswo, *Using a magnetometer to image a two-dimensional current distribution*, Journal of Applied Physics65,361–372(1989), [DOI](https://doi.org/10.1063/1.342549). [Author-institution full text](https://www.vanderbilt.edu/lsp/wp-content/uploads/sites/154/2026/08/jap-roth-using-89.pdf).

Logan Bishop-Van Horn, *current-reconstruction*, modern MIT-licensed implementation and deposited measured scanning-SQUID example,2024. The selected implementation uses explicit odd-grid frequency/origin corrections and a zero-current-DC convention; it is not claimed to be unchanged original code or software supplied by the1989 paper authors.
