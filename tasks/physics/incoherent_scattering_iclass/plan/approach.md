# Implemented Approach

## Bound research problem and scope

This task estimates an effective fluorescence image and an unknown intensity optical transfer function (OTF) from 150 experimentally recorded, independently speckle-illuminated camera frames. There are 160000 image samples, an equally large transfer-function grid, and 24 million observed intensity samples. Unknown illumination fields are integrated out statistically rather than supplied as calibration. A square virtual reflection matrix would contain 25.6 billion complex entries, approximately 205 GB in complex64. The shipped algorithm operates on a 160000-by-150 complex matrix and does not construct that square matrix.

The selected method is the memory-efficient I-CLASS phase kernel released with Weinberg, Sunray and Katz, Science Advances 10, eadl5218 (2024), followed by the per-frame amplitude compensation in Supplement S18. It is not microwave CC-CSI or a transmission-matrix pseudoinverse. The rejected microwave screening and the change of candidate are recorded privately and in validation. The released I-CLASS output routines have two discrepancies with physical spatial inversion and S18; these are explicitly corrected, independently tested and never described as unchanged original output reproduction.

The dataset is experimental Figure 3(D–F), scattering Run_1, 150 by 1200 by 1200, with a separately measured no-diffuser Run_0, 100 by 1200 by 1200, retained only for independent review. We bind a whole-field 400-by-400 Fourier resampling supported by upstream runCLASS. We do not claim to know the paper's undisclosed 400-pixel spatial crop or reproduce its exact figure. The public observations retain every illumination and the entire acquired field. A trial central spatial crop discarded extended PSF support and was rejected after original-code execution, not turned into a reference.

## Physical forward model and identifiable quantities

For spatial index r=(iy,ix), the measured intensity is

```math
I_m(r)=\sum_{r'}P(r-r')O(r')S_m(r')+\epsilon_m(r).
```

P is the intensity PSF, O is the nonnegative fluorescence distribution, S is the unknown random illumination intensity, m indexes 150 independent diffuser positions, and epsilon collects photon/read noise and model error. In the memory-effect/isoplanatic approximation, the same P applies to all object positions within the field. The discrete numerical model is circular after spatial apodization. This is a boundary approximation, not a physical periodic specimen. The independent forward-check function evaluates this convolution directly by spatial FFTs; it is not used as a known-calibration inverse shortcut.

The PSF's spatial integral defines an unknown throughput. Scaling O and inversely scaling P leaves the acquisition unchanged. A corresponding spatial shift of O and P is also unidentifiable. The outputs are therefore relative: image peak one, MTF peak one and a unit-modulus correction phasor. Absolute fluorophore concentration, transmission and object-plane micrometers cannot be recovered from the supplied calibration. The intensity OTF H=F2(P) is complex even though fluorescence detection is incoherent. Its phase is the phase of an intensity transfer function, not a coherent pupil wavefront.

The source's ideal covariance argument assumes spatially decorrelated illumination after temporal centering. If the illumination mean varies spatially, the recovered quantity includes this envelope. Specifically, for A=P diag(O) S and centered illumination S0, the approximation S0 S0^T approximately equals D, a diagonal matrix of local illumination variances. Consequently

```math
V=A_0A_0^T\approx P\,\mathrm{diag}(O)D\,\mathrm{diag}(O)P^T.
```

The effective object is O times sqrt(diag(D)), up to a global factor. Finite M=150, nonzero speckle-grain width, a finite detector field and anisoplanatism violate the exact diagonal model. These explain why a no-diffuser image is useful independent evidence but is not pixelwise ground truth for the recovered effective object. No hidden illumination or target-derived PSF is supplied publicly.

## Observation conversion, precision and coordinates

The original MATLAB v7.3 I_cam array is read using HDF5 and its reversed storage axes are restored. A stored (150,1200,1200) array is transposed to (illumination,camera row,camera column), corresponding to the original mat73 reader and its [2,0,1] permutation. Real observations are converted to float32 exactly as upstream does before its Fourier computation. Acquisition scalars remain verbatim in validation; stored image size 1200 overrides stale 1400 scalar size fields. Nominal detection wavelength 0.52 micrometers remains distinct from the paper's 525-nm/39-nm emission filter specification.

Each full camera frame undergoes an unnormalized 2D forward FFT, centered frequency cropping from 1200 to 400 samples along both spatial axes and an inverse FFT on the 400 grid. The crop is spectral, not spatial. Its inverse-size normalization yields a factor-nine DN amplitude compared with amplitude-preserving downsampling. This exact upstream convention is retained. It cancels from normalized MTF, phase direction and relative fluorescence, but must not silently be removed in parity checks. The processed real float32 array is the sole public observation key.

The first spatial coordinate increases down camera rows, the second right along columns. Forward FFT uses exp(-2*pi*i*k*r/N); inverse FFT divides by the number of spatial samples. The relative model uses detector-sampling coordinates. No absolute object-plane pitch is invented from objective magnification alone.

## Window, centering and rectangular matrix

On the released H=W=400 grid, define a symmetric Hann window w[j]=0.5*(1-cos(2*pi*j/(H-1))), j from zero through H-1. Endpoints are exactly zero. Multiply each frame successively by the vertical and horizontal windows, preserving upstream float32 multiplication order. Subtract each camera pixel's temporal mean across M frames. This produces centered frames X with units source FFT-resampled DN.

The source computes B=fftshift(FFT2(X)) with fftshift over all three axes. This also rotates frame order by 75. A frame permutation leaves covariance and RMS invariant; it is preserved for numerical parity. These preprocessing FFTs execute on CPU exactly as the oracle, followed by a transfer to CUDA when selected. Changing the preprocessing backend can alter a nonconvex phase trajectory through small roundoff changes.

The rectangular Fourier matrix R has shape (N,M), N=H*W=160000. Its row index is n=ix*H+iy. In code, R=B.permute(2,1,0).reshape(N,M). This column-major spatial convention is inverted explicitly before physical image formation. All phase iterates and matrices are complex64. Spatial windows and stored outputs are float32. CPU and CUDA have separately measured parity; no simulation is used in the bound reconstruction.

## Memory-efficient phase update

Initialize R0 to the measured rectangular matrix and the cumulative correction c0[n] to one. Each of T=250 iterations computes a covariance phase-alignment direction, takes its argument, and multiplies each row by the resulting unit phasor. There is no learned checkpoint, external PSF, optimizer library, training dataset or target-derived initialization.

Let C(A,B)=ifft2(fft2(A)*fft2(B))/(N*M) and K(A,B)=ifft2(fft2(A)*conj(fft2(B)))/(N*M) for an N-by-M first argument. These are circular convolution/correlation on the rectangular matrix axes, retaining the source's extra scale. A one-column second argument broadcasts in the frame-frequency dimension exactly as upstream. Define U as row reversal and L as column reversal. One iteration is

```math
b=C(\overline{UR},LR)_{:,M-1},\qquad
T=\mathrm{roll}_0\left[U K(UR,\overline b)\right],
```

where roll0 shifts the row axis by plus one and b is treated as a column for the correlation. The direction is z[n]=mean_m(conj(R[n,m])*T[n,m]); then q[n]=exp(i*arg(z[n])), c<-c*q and R<-diag(q)R. Updates are implemented by elementwise multiplication, never a dense diagonal matrix. The source's phase-only sign and reversal/roll conventions are preserved. At z=0, PyTorch angle gives zero and q=1. We use exactly 250 iterations, the original demonstration setting; no invented early-stop tolerance is substituted.

For independent validation, form the small dense virtual covariance V=R R^H and circular diagonal sums s[d]=sum_a V[(a+d) mod N,a]. Direct enumeration gives

```math
z[n]=\frac{1}{N^2M^3}\sum_a\overline{V[n,a]}s[(n-a)\bmod N].
```

This expression is checked against the FFT implementation on complex128 finite matrices. It validates conjugations, the phase sign, correlation reversal and the harmless positive normalization without importing an upstream operator. The bound task uses only the fast version. The released kernel uses circular flattened-frequency differences; the supplementary derivation also discusses full zero-padded convolution. We retain and identify the released periodic discretization rather than claiming the two boundary conditions are identical.

## MTF, output correction and image

The retained phase kernel uses the cyclic group Z_(H*W) after packing row
ix*H+iy, rather than the product group Z_H x Z_W of genuine two-dimensional
periodic shifts. A flattened displacement can carry across an iy boundary into
ix; that is not the same as independently wrapping a fixed 2D displacement on
both axes. Thus the dense virtual-matrix identity above establishes the released
flattened-circulant algorithm's algebra, not a complete 2D optical theorem or
equivalence to the supplement's padded optical construction. This is an explicit
reference-release indexing approximation. The independent physical convolution
tests use true 2D periodic transforms and do not erase that model limitation.
The 250-step core is retained, rather than silently changed to another operator.

The empirical MTF amplitude is h[n]=sqrt(sum_m |R[n,m]|^2 / max_j sum_m |R[j,m]|^2). Phase-only corrections do not change these powers, so it can be evaluated before the iterations. This is the Fourier covariance diagonal result S16/S17. It is an estimate from legitimate observations, not externally supplied calibration. It contains finite-sample noise and can be inaccurate in weak-signal bands.

The source ICLASS.RtoO applies ifft2 directly to an N-by-M matrix and then reshapes a row power into the image. That transforms the flattened spatial index and the illumination axis. A physical image requires restoring the H-by-W frequency grid and transforming its two spatial axes. An independent known-image round trip demonstrates the difference. The corrected implementation first reshapes R to (W,H,M), permutes to (M,H,W), then applies spatial ifftshift and inverse FFT.

The source visualization additionally divides the FFT of its already-combined RMS image by MTF+sigma. S18 instead divides each corrected frame spectrum by that quantity before spatial inversion and RMS combination. These operations do not commute. We preserve the original legacy outputs separately, implement S18 explicitly and verify it with an independent NumPy finite case. This is a documented correction, not an unchanged reproduction claim.

The fixed sigma=0.01 is the balanced example shown in Supplement Fig. S1, relative to peak-normalized MTF. It is not selected from a sweep against the hidden widefield image. For every corrected spectrum Bm, calculate Ym=IFFT2(ifftshift(Bm/(h+sigma))), then Oest=sqrt(mean_m |Ym|^2). Divide Oest by its maximum. Also compute the phase-only comparison by omitting h compensation; the same spatial packing and RMS apply. Both are actively computed fresh in the notebook. Cumulative c and h are unpacked to centered (H,W) maps using the identical inverse row convention.

## Observation consistency, metrics and limitations

The Judge has no solver imports. It reconstructs windowed, centered Fourier observations independently in NumPy/SciPy float64. It checks that submitted h agrees with the empirical MTF and that the image agrees with per-frame compensation using submitted c and h. Relative L2 errors have fixed nonzero numeric boundaries. It also evaluates J=sum_d |s[d]|^2/trace(V)^2 from the corrected rectangular matrix without building V. Normalization precedes fourth-order accumulation to avoid float32 overflow. This measures coherent-diagonal covariance accumulation under the model; it is not a complete camera-noise likelihood or recovery of the unknown illumination patterns.

The independent oracle has J=7.270196452047658, versus uncorrected approximately 6.572207. The threshold is 95 percent of the independently measured oracle value, which the uncorrected phase fails. An initial development test incorrectly assumed an unsupported factor-1.5 improvement; it was replaced by oracle parity and positive measured gain. This correction of a physical test assumption is recorded separately from unchanged Judge boundaries.

Image agreement is cosine NCC and dynamic-range NRMSE after a fixed integer circular translation chosen by maximum cross-correlation. Both images must already obey the peak-one output contract; no amplitude fitting, crop or inversion is performed. The hidden numeric reference is the independently corrected standard-method result, not the no-diffuser observation. Nonzero tolerances address numerical reproducibility; self-reference zero error is not used to require an impossible exact-zero threshold.

All three required outputs are evaluated: image reference/formation agreement, phase modulus and covariance coherence, and MTF observation agreement. Invalid shapes, wrong dtypes, complex intensity, NaN/Inf, missing keys, degenerate references and unavailable calibration are errors. Finite nonnegative but poor images fail quality. The independent widefield comparison is reported separately with shift and peak normalization; it diagnoses recovered bead structure and contrast rather than defining truth accuracy. Hann suppresses edge contributions, the illumination envelope differs, and the diffraction/sampling and exposure conventions differ. These residual limitations are shown without aesthetic background subtraction or invented missing comparisons.


## Current benchmark v2 acceptance

The v2 Judge retains independently evaluated image/MTF consistency and the already frozen covariance coherence threshold6.906686629445275, replacing reference-image similarity. No ground truth exists. A distinct smooth-phase candidate with its freshly computed consistent image passes; a random-phase self-consistent image fails covariance quality.31 unit tests passed. Phase-update250 iterations no longer define acceptance. See iclass_pytest.log.

Source reconstruction and historical comparison computations remain unchanged; the standalone Judge is independent and uses the current README/metrics contract.
